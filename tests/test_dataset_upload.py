"""Exercise dataset cleanup through the real upload callback in every workflow."""

import asyncio
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import binary_classification_classical as bcc
import regression_hybrid as rg
import regression_3d as rg3
import regression_classical as rgc


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, False),
    (bc3, bc3.binary_classification_3d_tab_content, True),
    (bcc, bcc.binary_classification_classical_tab_content, False),
    (rg, rg.regression_hybrid_tab_content, False),
    (rg3, rg3.regression_3d_tab_content, True),
    (rgc, rgc.regression_classical_tab_content, False),
]


@pytest.fixture(params=CASES, ids=[case[0].__name__ for case in CASES])
def upload_ui(request, tmp_path, monkeypatch):
    module, builder, is_3d = request.param
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path / "Data"))
    # File inputs require a path in Gradio's upload cache, just as browser uploads do.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(uploads))
    workdir = tmp_path / "working"
    workdir.mkdir()
    with gr.Blocks() as blocks:
        builder()
    callback = next(callback for callback in blocks.fns.values()
                    if getattr(callback.fn, "__name__", None) == "on_upload_dataset")
    session = SessionState(blocks)
    session[callback.inputs[1]._id] = str(workdir)

    def source(frame, name="source.csv"):
        path = uploads / name
        frame.to_csv(path, index=False)
        return path

    async def upload(path, *, options=None, targets="target", name="dataset", success=True):
        values = [getattr(component, "value", None) for component in callback.inputs]
        values[:5] = [{"path": str(path), "orig_name": path.name,
                       "meta": {"_type": "gradio.FileData"}},
                      None, name, "molecule", targets]
        if options is not None:
            values[-3:] = options
        for index, component in enumerate(callback.inputs):
            if isinstance(component, gr.State):
                values[index] = None
        response = await asyncio.wait_for(
            blocks.process_api(callback, values, state=session, session_hash="dataset-upload"), 20)
        assert len(response["data"]) == len(callback.outputs)
        outputs = dict(zip(callback.outputs, response["data"], strict=True))
        message = next(value for component, value in outputs.items() if isinstance(component, gr.HTML))
        assert ("color: green" if success else "color: red") in message, message
        preview = next(value for component, value in outputs.items()
                       if isinstance(component, gr.Dataframe))
        if success:
            preview = pd.DataFrame(preview["data"], columns=preview["headers"])
        return message, preview

    return SimpleNamespace(source=source, upload=upload, workdir=workdir, is_3d=is_3d)


def mixed_source():
    # Equivalent SMILES and salt forms deliberately carry different targets: keeping
    # the first molecule must also keep its original target, not a later duplicate's.
    return pd.DataFrame({
        "molecule": ["OCC", "CCO", "CC(=O)[O-].[Na+]", "CC(=O)[O-]",
                     "not-a-smiles", "[Na+]", "CCN", "CCC"],
        "target": [101, 102, 103, 104, 105, 106, None, 108],
    })


def assert_counts(message, imported, *, missing=0, invalid=0, carbonless=0, duplicates=0):
    removed = missing + invalid + carbonless + duplicates
    assert f"Imported {imported} molecules" in message
    assert f"removed {removed} rows" in message
    assert f"{missing} missing values" in message
    assert f"{invalid} invalid SMILES" in message
    assert f"{carbonless} without carbon" in message
    assert f"{duplicates} duplicates" in message


def test_upload_defaults_and_each_unchecked_option_save_aligned_rows(upload_ui):
    path = upload_ui.source(mixed_source())
    variants = [
        (None, ["CCO", "CC(=O)[O-]", "CCC"], [101, 103, 108], 2),
        ((False, True, True), ["OCC", "CC(=O)[O-]", "CCC"], [101, 103, 108], 2),
        ((True, False, True), ["CCO", "CCO", "CC(=O)[O-]", "CC(=O)[O-]", "CCC"],
         [101, 102, 103, 104, 108], 0),
        ((True, True, False), ["CCO", "CC(=O)[O-].[Na+]", "CC(=O)[O-]", "CCC"],
         [101, 103, 104, 108], 1),
        ((False, False, False), ["OCC", "CCO", "CC(=O)[O-].[Na+]", "CC(=O)[O-]", "CCC"],
         [101, 102, 103, 104, 108], 0),
    ]

    async def run():
        for options, smiles, targets, duplicates in variants:
            message, preview = await upload_ui.upload(path, options=options)
            expected = pd.DataFrame({"SMILES": smiles, "target": targets})
            saved = pd.read_csv(upload_ui.workdir / "dataset.csv")
            pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
            pd.testing.assert_frame_equal(preview, expected, check_dtype=False)
            assert_counts(message, len(expected), missing=1, invalid=1, carbonless=1,
                          duplicates=duplicates)

    asyncio.run(run())


def test_empty_and_all_invalid_uploads_preserve_existing_csv_and_preview(upload_ui):
    original = upload_ui.workdir / "dataset.csv"
    original.write_text("SMILES,target\nCCO,7\n", encoding="utf-8")
    before = original.read_bytes()
    empty = upload_ui.source(pd.DataFrame(columns=["molecule", "target"]), "empty.csv")
    invalid = upload_ui.source(pd.DataFrame({
        "molecule": ["not-a-smiles", "[Na+]", None], "target": [1, 2, 3]}), "invalid.csv")

    async def run():
        for source, counts in [(empty, {}), (invalid, dict(missing=1, invalid=1, carbonless=1))]:
            message, preview = await upload_ui.upload(source, success=False)
            assert_counts(message, 0, **counts)
            assert original.read_bytes() == before
            assert preview.get("__type__") == "update"
            assert "value" not in preview  # Preserve the displayed preview on failure.

    asyncio.run(run())


def test_upload_single_target_restriction_is_specific_to_3d(upload_ui):
    source = upload_ui.source(pd.DataFrame({
        "molecule": ["CCO", "CCN"], "target": [0, 1], "other": [1, 0]}))
    original = upload_ui.workdir / "dataset.csv"
    original.write_text("SMILES,target\nCCC,5\n", encoding="utf-8")
    before = original.read_bytes()

    async def run():
        message, preview = await upload_ui.upload(
            source, targets="target,other", success=not upload_ui.is_3d)
        if upload_ui.is_3d:
            assert "1 target" in message or "one target" in message
            assert original.read_bytes() == before
        else:
            expected = pd.DataFrame({"SMILES": ["CCO", "CCN"], "target": [0, 1], "other": [1, 0]})
            pd.testing.assert_frame_equal(pd.read_csv(original), expected)
            pd.testing.assert_frame_equal(preview, expected)
            assert_counts(message, 2)

    asyncio.run(run())


def test_prediction_only_upload_ignores_unselected_missing_targets(upload_ui):
    source = upload_ui.source(mixed_source())

    async def run():
        message, preview = await upload_ui.upload(source, targets="", name="prediction")
        expected = pd.DataFrame({"SMILES": ["CCO", "CC(=O)[O-]", "CCN", "CCC"]})
        pd.testing.assert_frame_equal(pd.read_csv(upload_ui.workdir / "prediction.csv"), expected)
        pd.testing.assert_frame_equal(preview, expected)
        assert_counts(message, 4, invalid=1, carbonless=1, duplicates=2)

    asyncio.run(run())
