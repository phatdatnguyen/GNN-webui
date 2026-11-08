"""Folder categories must agree between extraction and all six workflow selectors."""

import gradio as gr
import pytest
import torch

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import binary_classification_classical as bcc
import regression_hybrid as rg
import regression_3d as rg3
import regression_classical as rgc
import utils


@pytest.mark.parametrize("kind,prefix", [("graph", "GRAPH_"), ("fingerprint", "FP_")])
def test_feature_prefix_is_added_once_after_path_validation(kind, prefix):
    assert utils.prefixed_feature_directory(" custom ", kind) == prefix + "custom"
    assert utils.prefixed_feature_directory(prefix + "custom", kind) == prefix + "custom"
    for name in ("", "../escape", "/tmp/outside", "folder/child"):
        with pytest.raises(ValueError):
            utils.prefixed_feature_directory(name, kind)


def test_extractors_apply_prefixes_to_bare_names(tmp_path, progress):
    (tmp_path / "dataset.csv").write_text("SMILES,target\nCCO,1\n")
    for name in ("custom", "GRAPH_custom"):
        message = utils.extract_graphs(str(tmp_path), "dataset.csv", name,
                                       "MolGraphConvFeaturizer", torch.float32, progress)
        assert "color: green" in message, message
        assert (tmp_path / "GRAPH_custom" / "0.pt").is_file()
    for name in ("custom", "FP_custom"):
        message = utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", name,
                                                       "Morgan fingerprint", 2, 64, progress)
        assert "color: green" in message, message
        assert (tmp_path / "FP_custom" / "molecular_fingerprints.csv").is_file()
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_dir()) == ["FP_custom", "GRAPH_custom"]


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, "hybrid"),
    (rg, rg.regression_hybrid_tab_content, "hybrid"),
    (bc3, bc3.binary_classification_3d_tab_content, "3d"),
    (rg3, rg3.regression_3d_tab_content, "3d"),
    (bcc, bcc.binary_classification_classical_tab_content, "classical"),
    (rgc, rgc.regression_classical_tab_content, "classical"),
]


@pytest.mark.parametrize("module,builder,kind", CASES, ids=[case[0].__name__ for case in CASES])
def test_all_workflow_selectors_filter_artifact_categories(tmp_path, monkeypatch, module, builder, kind):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path / "Data"))
    workdir = tmp_path / "working"
    workdir.mkdir()
    (workdir / "dataset.csv").write_text("SMILES,target\nCCO,1\n")
    for directory in ("GRAPH_valid", "legacy_graphs", "FP_wrong", "MODEL_features"):
        (workdir / directory).mkdir()
        (workdir / directory / "0.pt").write_bytes(b"graph")
    for directory in ("FP_valid", "legacy_fingerprints", "GRAPH_wrong", "MODEL_features"):
        (workdir / directory).mkdir(exist_ok=True)
        (workdir / directory / "molecular_fingerprints.csv").write_text("SMILES,Bit_1\nCCO,1\n")
    for directory in ("MODEL_run_7", "unprefixed_run"):
        (workdir / directory).mkdir()
        for extension in ("ckpt", "joblib"):
            (workdir / directory / f"weights.{extension}").write_bytes(b"checkpoint")
            (workdir / f"root.{extension}").write_bytes(b"checkpoint")
    (workdir / "GRAPH_empty").mkdir()
    (workdir / "FP_empty").mkdir()
    (workdir / "GRAPH_link").symlink_to(workdir / "GRAPH_valid", target_is_directory=True)

    with gr.Blocks() as blocks:
        builder()
    callback = next(callback for callback in blocks.fns.values()
                    if getattr(callback.fn, "__name__", None) ==
                    ("refresh_directory" if kind == "classical" else "on_content_list_change"))
    if kind == "classical":
        updates = callback.fn(str(workdir), *([None] * 6))
    else:
        inputs = [str(workdir), "dataset", "GRAPH_valid"]
        if kind == "hybrid":
            inputs.append("FP_valid")
        inputs.append(None)
        updates = dict(zip(callback.outputs, callback.fn(*inputs), strict=True))
    graph_dropdowns = [component for component in callback.outputs
                       if isinstance(component, gr.Dropdown) and "graph directory" in component.label.lower()]
    fingerprint_dropdowns = [component for component in callback.outputs
                             if isinstance(component, gr.Dropdown) and "fingerprint directory" in component.label.lower()]
    for component in graph_dropdowns:
        assert updates[component]["choices"] == ["GRAPH_valid"]
    for component in fingerprint_dropdowns:
        assert updates[component]["choices"] == ["FP_valid"]
    assert len(graph_dropdowns) == (0 if kind == "classical" else 2)
    assert len(fingerprint_dropdowns) == (0 if kind == "3d" else 2)
    checkpoint_selector = next(component for component in callback.outputs
                               if isinstance(component, gr.Dropdown)
                               and component.label in ("Checkpoint file", "Saved model"))
    suffix = "joblib" if kind == "classical" else "ckpt"
    assert updates[checkpoint_selector]["choices"] == [f"MODEL_run_7/weights.{suffix}"]
    if kind == "hybrid":
        text = "\n".join(str(component.value) for component in blocks.blocks.values()
                         if isinstance(component, gr.Markdown))
        assert all(label in text for label in ("Graph Encoder", "Fingerprint Encoder", "Prediction Head"))
        assert not any(label in text.lower() for label in ("gcn block", "mlp block", "predictor block"))
