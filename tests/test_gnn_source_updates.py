"""Source mutations invalidate captured splits without discarding prediction uploads."""

import asyncio
import inspect
from types import SimpleNamespace

import gradio as gr
import pytest
from gradio.state_holder import SessionState

from gnn_run_ui import bind_artifact_update


class CaptureButton:
    def click(self, fn, inputs, outputs, **kwargs):
        return SimpleNamespace(fn=fn, inputs=inputs, outputs=outputs, **kwargs)


@pytest.fixture
def processed_dataset(tmp_path):
    return SimpleNamespace(split_context={"working_directory": str(tmp_path),
                                           "dataset_file": "training.csv"},
                           feature_sources={"graph_directory": "GRAPH_graphs",
                                            "fingerprint_directory": "FP_fingerprints"})


@pytest.mark.parametrize("name,success,same_directory,reset", [
    ("training", True, True, True), (" training ", True, True, True),
    ("prediction", True, True, False), ("training", False, True, False),
    ("training", True, False, False),
])
def test_upload_resets_only_successful_training_source_replacement(
        processed_dataset, name, success, same_directory, reset):
    calls = []
    status, preview, contents, dataset_state, analysis_state = [object() for _ in range(5)]
    inputs = [object() for _ in range(8)]

    def on_upload_dataset(file, workdir, dataset_name, smiles, targets,
                          canonicalize_smiles=True, remove_duplicates=True, remove_salt=True):
        calls.append((canonicalize_smiles, remove_duplicates, remove_salt))
        return ("<span style='color: green;'>Imported</span>" if success else
                "<span style='color: red;'>Invalid upload</span>"), "fresh preview", "fresh contents"

    callback = bind_artifact_update(
        CaptureButton(), on_upload_dataset, inputs, [status, preview, contents],
        dataset_state=dataset_state, reset_updates={dataset_state: None, analysis_state: None,
                                                  preview: "cleared preview", contents: "cleared contents"},
        kind="upload", concurrency_id="shared-tab")
    assert callback.inputs[5] is dataset_state and callback.inputs[-3:] == inputs[-3:]
    assert list(inspect.signature(callback.fn).parameters)[-3:] == [
        "canonicalize_smiles", "remove_duplicates", "remove_salt"]
    assert callback.fn.__name__ == "on_upload_dataset" and callback.concurrency_id == "shared-tab"
    directory = processed_dataset.split_context["working_directory"]
    result = callback.fn("source.csv", directory if same_directory else directory + "/other",
                         name, "SMILES", "target", processed_dataset, False, True, False)
    assert calls == [(False, True, False)]
    assert result[preview] == "fresh preview" and result[contents] == "fresh contents"
    assert (analysis_state in result) is reset
    assert (dataset_state in result) is reset


@pytest.mark.parametrize("kind,feature_directory", [("graphs", "graphs"),
                                                   ("fingerprints", "fingerprints")])
@pytest.mark.parametrize("same_source,same_features,success,reset", [
    (True, False, True, True), (False, True, True, True), (False, False, True, False),
    (True, False, False, True), (False, True, False, True), (False, False, False, False),
])
def test_extraction_resets_matching_dataset_or_feature_directory_even_after_failure(
        processed_dataset, kind, feature_directory, same_source, same_features, success, reset):
    status, contents, dataset_state, analysis_state = [object() for _ in range(4)]
    progress = gr.Progress()

    def extract(workdir, source, directory, progress=gr.Progress()):
        assert progress is expected_progress
        return "color: green" if success else "color: red", "fresh contents"

    expected_progress = progress
    callback = bind_artifact_update(
        CaptureButton(), extract, [object(), object(), object()], [status, contents],
        dataset_state=dataset_state, reset_updates={dataset_state: None, analysis_state: None}, kind=kind)
    assert list(inspect.signature(callback.fn).parameters)[-2:] == ["processed_dataset", "progress"]
    result = callback.fn(processed_dataset.split_context["working_directory"],
                         "training.csv" if same_source else "prediction.csv",
                         feature_directory if same_features else "prediction_features", processed_dataset, progress)
    assert result[contents] == "fresh contents"
    assert (analysis_state in result) is reset
    assert (dataset_state in result) is reset


def test_extraction_state_and_progress_survive_real_gradio_postprocessing(processed_dataset):
    seen = []

    def extract(workdir, source, directory, progress=gr.Progress()):
        seen.append(isinstance(progress, gr.Progress))
        return "<span style='color: green;'>Extracted</span>", ["training.csv", directory]

    with gr.Blocks() as blocks:
        button = gr.Button()
        inputs = [gr.Textbox(processed_dataset.split_context["working_directory"]),
                  gr.Textbox("training.csv"), gr.Textbox("graphs")]
        status, contents = gr.HTML(), gr.JSON()
        dataset_state, analysis_state = gr.State(), gr.State()
        export_button = gr.Button(interactive=True)
        bind_artifact_update(button, extract, inputs, [status, contents], dataset_state=dataset_state,
                             reset_updates={dataset_state: None, analysis_state: None,
                                            export_button: gr.update(interactive=False)},
                             kind="graphs", concurrency_id="shared-tab")
    callback = next(iter(blocks.fns.values()))
    session = SessionState(blocks)
    session[dataset_state._id] = processed_dataset
    session[analysis_state._id] = {"old": "analysis"}

    async def run():
        return await asyncio.wait_for(blocks.process_api(
            callback, [component.value for component in inputs] + [None],
            state=session, session_hash="source-update"), 20)

    response = asyncio.run(run())
    assert seen == [True]
    assert "color: green" in response["data"][0]
    assert session[dataset_state._id] is None and session[analysis_state._id] is None
    assert response["data"][-1]["interactive"] is False
