"""Shared data controls and classical graph extraction keep workflow state coherent."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest
import torch
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import binary_classification_classical as bcc
import regression_hybrid as rg
import regression_3d as rg3
import regression_classical as rgc
from classical_ui import ClassicalWorkflow, build_classical_tab
import utils


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, False, False),
    (bc3, bc3.binary_classification_3d_tab_content, True, False),
    (bcc, bcc.binary_classification_classical_tab_content, False, True),
    (rg, rg.regression_hybrid_tab_content, False, False),
    (rg3, rg3.regression_3d_tab_content, True, False),
    (rgc, rgc.regression_classical_tab_content, False, True),
]


def _callback(blocks, name):
    return next(callback for callback in blocks.fns.values()
                if getattr(callback.fn, "__name__", None) == name)


def _ancestor(component, component_type):
    parent = component.parent
    while parent is not None:
        if isinstance(parent, component_type):
            return parent
        parent = parent.parent
    return None


@pytest.mark.parametrize("module,builder,is_3d,is_classical", CASES,
                         ids=[case[0].__name__ for case in CASES])
def test_feature_controls_offer_graph_extraction_without_changing_model_inputs(
        tmp_path, monkeypatch, module, builder, is_3d, is_classical):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    with gr.Blocks() as blocks:
        builder()
    graph = _callback(blocks, "on_extract_graphs")
    graph_button = next(component for component in blocks.blocks.values()
                        if isinstance(component, gr.Button) and component.value == "Extract graphs")
    assert graph_button.interactive is False
    extraction_section = _ancestor(graph_button, gr.Accordion)
    assert extraction_section is not None
    assert graph.inputs[2].value.startswith("GRAPH_")
    if is_3d:
        labels = {component.label for component in graph.inputs if not isinstance(component, gr.State)}
        assert {"Number of conformers", "Force field"} <= labels
        assert graph.inputs[4].value == "MMFF"
    else:
        featurizer = graph.inputs[3]
        assert featurizer.label == "Graph featurizer"
        assert [value for _, value in featurizer.choices] == [
            "MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"]
        fingerprints = _callback(blocks, "on_extract_fingerprints" if is_classical
                                 else "on_extract_molecule_fingerprints")
        fingerprint_button = next(component for component in blocks.blocks.values()
                                  if isinstance(component, gr.Button)
                                  and component.value == "Extract molecular fingerprints")
        assert _ancestor(fingerprint_button, gr.Accordion) is extraction_section
        assert fingerprints.inputs[2].value == "FP_fingerprints"
        assert fingerprints.inputs[3].value == "Morgan fingerprint"
    if is_classical:
        # Sharing extraction tools must not turn the classical estimator into a GNN.
        processing = _callback(blocks, "on_process_data")
        assert processing.inputs[2].label == "Training fingerprint directory"
        assert not any("graph" in str(getattr(component, "label", "")).lower()
                       for component in processing.inputs)
        assert isinstance(graph.inputs[4], gr.State) and graph.inputs[4].value == torch.float32


@pytest.fixture(params=["classification", "regression"])
def classical_graph_ui(request, tmp_path):
    workflow = ClassicalWorkflow(request.param, lambda: str(tmp_path))
    with gr.Blocks() as blocks:
        build_classical_tab(workflow, request.param)
    _, path = workflow.on_open_working_directory("experiment")
    workdir = Path(path)
    pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC"], "target": [0, 1, 0]}).to_csv(
        workdir / "training.csv", index=False)
    pd.DataFrame({"SMILES": ["CCCC", "CCCl"]}).to_csv(workdir / "prediction.csv", index=False)
    session = SessionState(blocks)
    values = {component._id: getattr(component, "value", None) for component in blocks.blocks.values()}
    properties = {}
    graph = _callback(blocks, "on_extract_graphs")
    fit = _callback(blocks, "on_fit")
    similarity = _callback(blocks, "on_calculate_similarity")
    processed_state, model_state = fit.inputs[0], fit.inputs[3]
    similarity_state = next(component for component in similarity.outputs if isinstance(component, gr.State))
    tracked_states = [processed_state, model_state, similarity_state, similarity.inputs[0]]
    sentinels = [SimpleNamespace(identity=index) for index in range(len(tracked_states))]

    def prepare_active_run():
        for state, sentinel in zip(tracked_states, sentinels):
            session[state._id] = sentinel
        session[graph.inputs[5]._id] = (str(workdir), "training.csv", "FP_training")

    async def invoke(callback, overrides=None):
        for index, value in (overrides or {}).items():
            component = callback.inputs[index]
            if isinstance(component, gr.State):
                session[component._id] = value
            else:
                values[component._id] = value
        inputs = [None if isinstance(component, gr.State) else values[component._id]
                  for component in callback.inputs]
        response = await asyncio.wait_for(blocks.process_api(
            callback, inputs, state=session, session_hash=f"graph-{request.param}"), 20)
        assert len(response["data"]) == len(callback.outputs)
        for component, result in zip(callback.outputs, response["data"]):
            if isinstance(component, gr.State):
                continue
            if isinstance(result, dict) and result.get("__type__") == "update":
                properties.setdefault(component._id, {}).update(result)
                if "value" in result:
                    values[component._id] = result["value"]
            else:
                values[component._id] = result
        # Execute only the declared continuation, just as the browser follows .then.
        # Calling refresh_directory by name would hide a missing event connection.
        for continuation in blocks.fns.values():
            if continuation.trigger_after == callback._id:
                await invoke(continuation)
        return response

    async def open_directory():
        await invoke(_callback(blocks, "on_open_working_directory"), {0: "experiment"})

    return SimpleNamespace(blocks=blocks, session=session, values=values, properties=properties,
                           graph=graph, workdir=workdir, tracked_states=tracked_states,
                           sentinels=sentinels, prepare_active_run=prepare_active_run,
                           invoke=invoke, open_directory=open_directory)


def test_classical_graph_extraction_refreshes_contents_and_invalidates_only_training_source(classical_graph_ui):
    ui = classical_graph_ui

    async def run():
        await ui.open_directory()
        ui.prepare_active_run()
        for filename, folder, smiles, affects_training in [
            ("prediction.csv", "prediction_graphs", ["CCCC", "CCCl"], False),
            ("training.csv", "GRAPH_training_graphs", ["CCO", "CCN", "CCC"], True),
        ]:
            response = await ui.invoke(ui.graph, {1: filename, 2: folder})
            assert "color: green" in response["data"][0], response["data"][0]
            output = ui.workdir / (folder if folder.startswith("GRAPH_") else f"GRAPH_{folder}")
            assert utils.read_graph_manifest(output) == smiles
            assert utils.list_graph_files(output) == list(range(len(smiles)))
            for index, expected_smiles in enumerate(smiles):
                graph = torch.load(output / f"{index}.pt", weights_only=False)
                assert graph.smiles == expected_smiles
                assert graph.x.dtype == graph.edge_attr.dtype == torch.float32
                assert graph.edge_index.dtype == torch.long
            contents = next(component for component in ui.blocks.blocks.values()
                            if isinstance(component, gr.Dataframe)
                            and component.label == "Contents in Working Directory")
            assert any(row[0] == output.name for row in ui.values[contents._id]["data"])
            for state, sentinel in zip(ui.tracked_states, ui.sentinels):
                assert ui.session[state._id] is (None if affects_training else sentinel)
            assert pd.read_csv(ui.workdir / filename)["SMILES"].tolist() == smiles

    asyncio.run(run())


@pytest.mark.parametrize("training_source", [False, True])
def test_classical_graph_extraction_errors_are_readable_and_preserve_unrelated_runs(
        classical_graph_ui, monkeypatch, training_source):
    ui = classical_graph_ui

    def fail_extraction(*args, **kwargs):
        raise OSError("storage <unavailable>")

    monkeypatch.setattr(utils, "extract_graphs", fail_extraction)

    async def run():
        await ui.open_directory()
        ui.prepare_active_run()
        filename = "training.csv" if training_source else "prediction.csv"
        response = await ui.invoke(ui.graph, {1: filename, 2: "failed_graphs"})
        assert "color: red" in response["data"][0]
        assert "storage &lt;unavailable&gt;" in response["data"][0]
        assert not (ui.workdir / "GRAPH_failed_graphs").exists()
        for state, sentinel in zip(ui.tracked_states, ui.sentinels):
            assert ui.session[state._id] is (None if training_source else sentinel)

    asyncio.run(run())
