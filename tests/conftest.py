"""Shared fixtures for the GNN web UI test suite.

Feature extraction (graphs, fingerprints, conformers) is the slow part, so it is done once
per session in `*_workdir_template` and copied into a fresh directory for each test that
needs to mutate it.
"""
import os
import shutil
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402


# Small, quick-to-featurise molecules. All contain carbon so none are filtered out.
SMILES = [
    "CCO", "c1ccccc1", "CCN", "CC(=O)O", "CCCC", "CCCCO", "CCOC", "c1ccncc1",
    "CC(C)O", "CCCCCC", "C1CCCCC1", "CC(=O)N", "CCS", "CC#N", "CCCl", "c1ccc(O)cc1",
]
# Tiny molecules for the 3D tests - conformer generation is the slowest thing in the suite.
SMILES_3D = ["CCO", "CCN", "CCC", "CCS", "CC=O", "CCCl", "CC(C)C", "CCBr"]

BINARY_LABELS = [i % 2 for i in range(len(SMILES))]
REGRESSION_VALUES = [float(i) * 1.7 + 0.5 for i in range(len(SMILES))]


class FakeProgress:
    """Stand-in for gr.Progress(); the real one needs a live Gradio request context."""

    def tqdm(self, iterable, **kwargs):
        return iterable


@pytest.fixture
def progress():
    return FakeProgress()


def _write_source_csv(directory, smiles, targets, target_column):
    path = os.path.join(directory, "source.csv")
    pd.DataFrame({"smiles": smiles, target_column: targets}).to_csv(path, index=False)
    return path


def _build_2d_workdir(directory, targets, target_column):
    source = _write_source_csv(directory, SMILES, targets, target_column)
    status, df = utils.process_dataset_file(source, directory, "dataset", "smiles", target_column)
    assert df is not None, status
    status = utils.extract_graphs(directory, "dataset.csv", "graphs",
                                  "MolGraphConvFeaturizer", torch.float32, FakeProgress())
    assert "green" in status, status
    status = utils.extract_molecule_fingerprints(directory, "dataset.csv", "fps",
                                                 "Morgan fingerprint", 2, 128, FakeProgress())
    assert "green" in status, status
    return directory


@pytest.fixture(scope="session")
def classification_workdir_template(tmp_path_factory):
    return _build_2d_workdir(str(tmp_path_factory.mktemp("cls_template")), BINARY_LABELS, "target")


@pytest.fixture(scope="session")
def regression_workdir_template(tmp_path_factory):
    return _build_2d_workdir(str(tmp_path_factory.mktemp("reg_template")), REGRESSION_VALUES, "value")


@pytest.fixture(scope="session")
def workdir_3d_template(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("wd3d_template"))
    source = _write_source_csv(directory, SMILES_3D, [i % 2 for i in range(len(SMILES_3D))], "target")
    status, df = utils.process_dataset_file(source, directory, "dataset", "smiles", "target", True)
    assert df is not None, status
    status = utils.extract_3d_graphs(directory, "dataset.csv", "g3d", 3, "MMFF",
                                     torch.float32, FakeProgress())
    assert "green" in status, status
    return directory


def _fresh_copy(template, tmp_path, name):
    target = str(tmp_path / name)
    shutil.copytree(template, target)
    return target


@pytest.fixture
def classification_workdir(classification_workdir_template, tmp_path):
    """A writable copy of the classification working directory."""
    return _fresh_copy(classification_workdir_template, tmp_path, "cls")


@pytest.fixture
def regression_workdir(regression_workdir_template, tmp_path):
    return _fresh_copy(regression_workdir_template, tmp_path, "reg")


@pytest.fixture
def workdir_3d(workdir_3d_template, tmp_path):
    return _fresh_copy(workdir_3d_template, tmp_path, "wd3d")


@pytest.fixture
def empty_workdir(tmp_path):
    directory = str(tmp_path / "empty")
    os.makedirs(directory)
    return directory


# --------------------------------------------------------------------------------------
# Argument builders for the Gradio callbacks, which take long positional parameter lists.
# --------------------------------------------------------------------------------------

CREATE_MODEL_DEFAULTS = dict(
    gcn_n_hiddens=32, gcn_num_layers=2, gcn_dropout=0.0, gcn_n_outputs=16,
    graph_sage_n_hiddens=32, graph_sage_num_layers=2, graph_sage_dropout=0.0, graph_sage_n_outputs=16,
    gin_n_hiddens=32, gin_num_layers=2, gin_dropout=0.0, gin_n_outputs=16,
    gat_n_hiddens=32, gat_num_layers=2, gat_dropout=0.0, gat_n_outputs=16,
    edgecnn_n_hiddens=32, edgecnn_num_layers=2, edgecnn_dropout=0.0, edgecnn_n_outputs=16,
    attentivefp_n_hiddens=32, attentivefp_num_layers=2, attentivefp_num_timesteps=2,
    attentivefp_dropout=0.0, attentivefp_n_outputs=16,
    customgcn_convolutional_layer_name="GCNConv", customgcn_n_hiddens=32, customgcn_n_layers=2,
    customgcn_n_heads=2, customgcn_n_outputs=16,
    mlp_n_hiddens=32, mlp_n_layers=2, mlp_n_outputs=16,
    predictor_n_hiddens=32, predictor_n_layers=1,
)

_CREATE_MODEL_ORDER = [
    "gcn_n_hiddens", "gcn_num_layers", "gcn_dropout", "gcn_n_outputs",
    "graph_sage_n_hiddens", "graph_sage_num_layers", "graph_sage_dropout", "graph_sage_n_outputs",
    "gin_n_hiddens", "gin_num_layers", "gin_dropout", "gin_n_outputs",
    "gat_n_hiddens", "gat_num_layers", "gat_dropout", "gat_n_outputs",
    "edgecnn_n_hiddens", "edgecnn_num_layers", "edgecnn_dropout", "edgecnn_n_outputs",
    "attentivefp_n_hiddens", "attentivefp_num_layers", "attentivefp_num_timesteps",
    "attentivefp_dropout", "attentivefp_n_outputs",
    "customgcn_convolutional_layer_name", "customgcn_n_hiddens", "customgcn_n_layers",
    "customgcn_n_heads", "customgcn_n_outputs",
    "mlp_n_hiddens", "mlp_n_layers", "mlp_n_outputs",
    "predictor_n_hiddens", "predictor_n_layers",
]


def create_model_args(dataset, gnn_model_tab, variance_threshold=None, pca=None, **overrides):
    """Build the positional argument list for on_create_model in the 2D tabs."""
    unknown = set(overrides) - set(CREATE_MODEL_DEFAULTS)
    assert not unknown, f"unknown create_model override(s): {sorted(unknown)}"
    values = {**CREATE_MODEL_DEFAULTS, **overrides}
    head = [dataset, gnn_model_tab, "cpu", torch.float32, 0, variance_threshold, pca]
    return head + [values[name] for name in _CREATE_MODEL_ORDER]


CREATE_MODEL_3D_DEFAULTS = dict(
    schnet_n_hiddens=32, schnet_n_filters=32, schnet_n_interactions=2, schnet_n_gaussians=25,
    schnet_cutoff=10.0, schnet_max_num_neighbors=32, schnet_readout="add", schnet_dipole=False,
    dimenetplusplus_n_hiddens=32, dimenetplusplus_n_blocks=2, dimenetplusplus_int_emb_size=8,
    dimenetplusplus_basis_emb_size=8, dimenetplusplus_out_emb_channels=8,
    dimenetplusplus_n_spherical=7, dimenetplusplus_n_radial=6, dimenetplusplus_cutoff=10.0,
    dimenetplusplus_max_num_neighbors=32, dimenetplusplus_envelope_exponent=5,
    dimenetplusplus_n_output_layers=3,
    visnet_n_hiddens=32, visnet_n_heads=4, visnet_n_layers=2, visnet_num_rbf=16,
    visnet_trainable_rbf=False, visnet_max_z=100, visnet_cutoff=10.0,
    visnet_max_num_neighbors=32, visnet_vertex=False,
)

_CREATE_MODEL_3D_ORDER = list(CREATE_MODEL_3D_DEFAULTS)


def create_model_3d_args(dataset, gnn_model_tab, **overrides):
    """Build the positional argument list for on_create_model in the 3D tabs."""
    unknown = set(overrides) - set(CREATE_MODEL_3D_DEFAULTS)
    assert not unknown, f"unknown create_model_3d override(s): {sorted(unknown)}"
    values = {**CREATE_MODEL_3D_DEFAULTS, **overrides}
    head = [dataset, gnn_model_tab, "cpu", torch.float32, 0]
    return head + [values[name] for name in _CREATE_MODEL_3D_ORDER]


def is_ok(status_html):
    """The callbacks report success/failure by colouring an HTML status string."""
    return "color: green" in status_html


def is_error(status_html):
    return "color: red" in status_html
