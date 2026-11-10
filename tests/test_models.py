"""Tests for the model zoo in models.py.

Every architecture offered in the UI is built and run here, at every layer count the
sliders allow, because a wrong output dimension only shows up at forward time.
"""
import pytest
import torch

import models
from models import (
    MLPModel, GCNModel, GraphSAGEModel, GINModel, GATModel, EdgeCNNModel, AttentiveFPModel,
)

N_IN, N_HID, N_OUT, EDGE_DIM, HEADS = 30, 32, 16, 11, 2
MLP = (64, 32, 2, 8)          # n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs
PRED = (32, 1)                # n_predictor_hiddens, n_predictor_layers
N_TARGETS = 1

# Layers whose forward() takes edge_attr; mirrors the `need_edge_attr` list in the tabs.
EDGE_ATTR_LAYERS = {
    "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv",
    "TransformerConv", "GeneralConv",
}

ALL_CUSTOM_LAYERS = [
    "GCNConv", "SAGEConv", "SGConv", "ClusterGCNConv", "GraphConv", "ChebConv", "LEConv",
    "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv",
    "ResGatedGraphConv", "GATConv", "GATv2Conv", "SuperGATConv", "TransformerConv", "GeneralConv",
]


@pytest.fixture
def graph():
    """A two-molecule batch, so pooling and the batch vector are actually exercised."""
    torch.manual_seed(0)
    n_nodes = 12
    # A PyG batch contains disjoint graphs: bonds never connect different molecules.
    edge_index = torch.cat([
        torch.randint(0, 6, (2, 15)),
        torch.randint(0, 6, (2, 15)) + 6,
    ], dim=1)
    return dict(
        x=torch.randn(n_nodes, N_IN),
        edge_index=edge_index,
        edge_attr=torch.randn(30, EDGE_DIM),
        batch=torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)]),
        fp=torch.randn(2, MLP[0]),
    )


def build_custom(layer_name, n_layers, n_gcn_outputs=N_OUT, n_mlp_outputs=MLP[3]):
    mlp = (MLP[0], MLP[1], MLP[2], n_mlp_outputs)
    plain = (N_IN, N_HID, n_layers, n_gcn_outputs, *mlp, *PRED, N_TARGETS)
    with_edge = (N_IN, N_HID, n_layers, n_gcn_outputs, EDGE_DIM, *mlp, *PRED, N_TARGETS)
    heads_edge = (N_IN, N_HID, n_layers, HEADS, n_gcn_outputs, EDGE_DIM, *mlp, *PRED, N_TARGETS)
    heads_plain = (N_IN, N_HID, n_layers, HEADS, n_gcn_outputs, *mlp, *PRED, N_TARGETS)

    ctors = {
        "GCNConv": (models.GCNConvModel, plain),
        "SAGEConv": (models.SAGEConvModel, plain),
        "SGConv": (models.SGConvModel, plain),
        "ClusterGCNConv": (models.ClusterGCNConvModel, plain),
        "GraphConv": (models.GraphConvModel, plain),
        "ChebConv": (models.ChebConvModel, plain),
        "LEConv": (models.LEConvModel, plain),
        "EGConv": (models.EGConvModel, plain),
        "MFConv": (models.MFConvModel, plain),
        "FeaStConv": (models.FeaStConvModel, plain),
        "TAGConv": (models.TAGConvModel, plain),
        "ARMAConv": (models.ARMAConvModel, plain),
        "FiLMConv": (models.FiLMConvModel, plain),
        "PDNConv": (models.PDNConvModel,
                    (N_IN, N_HID, n_layers, n_gcn_outputs, EDGE_DIM, N_HID, *mlp, *PRED, N_TARGETS)),
        "GENConv": (models.GENConvModel, with_edge),
        "ResGatedGraphConv": (models.ResGatedGraphConvModel, with_edge),
        "GATConv": (models.GATConvModel, heads_edge),
        "GATv2Conv": (models.GATv2ConvModel, heads_edge),
        "SuperGATConv": (models.SuperGATConvModel, heads_plain),
        "TransformerConv": (models.TransformerConvModel, heads_edge),
        "GeneralConv": (models.GeneralConvModel, heads_edge),
    }
    ctor, args = ctors[layer_name]
    return ctor(*args)


def run(model, layer_name, graph, num_graphs=None):
    model.eval()
    options = models.graph_count_kwargs(model, num_graphs) if num_graphs is not None else {}
    with torch.no_grad():
        if layer_name in EDGE_ATTR_LAYERS:
            return model(graph["x"], graph["edge_index"], graph["edge_attr"],
                         graph["batch"], graph["fp"], **options)
        return model(graph["x"], graph["edge_index"], graph["batch"], graph["fp"], **options)


# ------------------------------------------------------------------- custom conv layers

@pytest.mark.parametrize("layer_name", ALL_CUSTOM_LAYERS)
@pytest.mark.parametrize("n_layers", [1, 2, 3, 6])
def test_every_custom_layer_builds_and_runs(layer_name, n_layers, graph):
    """Regression: with n_layers=1 (the slider minimum) eight of these skipped the final
    projection to n_gcn_outputs and blew up with a shape mismatch at forward time."""
    model = build_custom(layer_name, n_layers)
    out = run(model, layer_name, graph)
    assert out.shape == (2, N_TARGETS)
    assert torch.isfinite(out).all()
    # Supplying the CPU loader's graph count must not alter pooling or weights.
    torch.testing.assert_close(run(model, layer_name, graph, num_graphs=2), out)


@pytest.mark.parametrize("layer_name", ALL_CUSTOM_LAYERS)
def test_custom_layers_work_without_the_fingerprint_branch(layer_name, graph):
    model = build_custom(layer_name, 2, n_mlp_outputs=0)
    model.eval()
    with torch.no_grad():
        if layer_name in EDGE_ATTR_LAYERS:
            out = model(graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], None)
        else:
            out = model(graph["x"], graph["edge_index"], graph["batch"], None)
    assert out.shape == (2, N_TARGETS)


@pytest.mark.parametrize("layer_name", ALL_CUSTOM_LAYERS)
def test_custom_layers_reject_zero_total_output_width(layer_name):
    with pytest.raises(ValueError, match="cannot be 0"):
        build_custom(layer_name, 2, n_gcn_outputs=0, n_mlp_outputs=0)


def test_chebconv_gets_a_filter_size(graph):
    """Regression: ChebConv's required K argument was missing, so it failed to construct."""
    model = build_custom("ChebConv", 2)
    assert model.graph_encoder[0].__class__.__name__ == "ChebConv"
    assert run(model, "ChebConv", graph).shape == (2, N_TARGETS)
    assert models.CHEB_CONV_K >= 1


# --------------------------------------------------------------------- top level models

TOP_LEVEL = {
    "GCN": (lambda: GCNModel(N_IN, N_HID, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS), False),
    "GraphSAGE": (lambda: GraphSAGEModel(N_IN, N_HID, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS), False),
    "GIN": (lambda: GINModel(N_IN, N_HID, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS), False),
    "EdgeCNN": (lambda: EdgeCNNModel(N_IN, N_HID, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS), False),
    "GAT": (lambda: GATModel(N_IN, N_HID, EDGE_DIM, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS), True),
    "AttentiveFP": (lambda: AttentiveFPModel(N_IN, N_HID, EDGE_DIM, 2, 2, N_OUT, 0.0,
                                             *MLP, *PRED, N_TARGETS), True),
}


@pytest.mark.parametrize("name", list(TOP_LEVEL))
def test_top_level_models_build_and_run(name, graph):
    ctor, needs_edge_attr = TOP_LEVEL[name]
    model = ctor()
    model.eval()
    with torch.no_grad():
        if needs_edge_attr:
            args = (graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], graph["fp"])
        else:
            args = (graph["x"], graph["edge_index"], graph["batch"], graph["fp"])
        out = model(*args)
        torch.testing.assert_close(model(*args, **models.graph_count_kwargs(model, 2)), out)
    assert out.shape == (2, N_TARGETS)
    assert torch.isfinite(out).all()


def test_pooling_separates_the_graphs_in_a_batch(graph):
    """The batch vector must reach global_mean_pool: one row of output per molecule,
    and changing one molecule's features must not change the other's prediction."""
    torch.manual_seed(0)
    model = GCNModel(N_IN, N_HID, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS)
    model.eval()
    with torch.no_grad():
        first = model(graph["x"], graph["edge_index"], graph["batch"], graph["fp"])

        perturbed = graph["x"].clone()
        perturbed[6:] += 10.0                     # only the second molecule's nodes
        second = model(perturbed, graph["edge_index"], graph["batch"], graph["fp"])

    assert first.shape == (2, N_TARGETS)
    torch.testing.assert_close(first[0], second[0])
    assert not torch.allclose(first[1], second[1])


# --------------------------------------------------------------------- edge feature use

EDGE_AWARE_MODELS = [
    ("GAT", lambda: GATModel(N_IN, N_HID, EDGE_DIM, 2, N_OUT, 0.0, *MLP, *PRED, N_TARGETS)),
    ("AttentiveFP", lambda: AttentiveFPModel(N_IN, N_HID, EDGE_DIM, 2, 2, N_OUT, 0.0,
                                             *MLP, *PRED, N_TARGETS)),
    ("GATConv", lambda: build_custom("GATConv", 2)),
    ("GATv2Conv", lambda: build_custom("GATv2Conv", 2)),
    ("TransformerConv", lambda: build_custom("TransformerConv", 2)),
    ("GENConv", lambda: build_custom("GENConv", 2)),
    ("ResGatedGraphConv", lambda: build_custom("ResGatedGraphConv", 2)),
    ("PDNConv", lambda: build_custom("PDNConv", 2)),
    ("GeneralConv", lambda: build_custom("GeneralConv", 2)),
]


@pytest.mark.parametrize("name,ctor", EDGE_AWARE_MODELS, ids=[n for n, _ in EDGE_AWARE_MODELS])
def test_models_declared_edge_aware_actually_use_edge_features(name, ctor, graph):
    """Regression: GAT and the custom GATConv were never given edge_dim, so PyG silently
    discarded the bond features the UI was feeding them."""
    torch.manual_seed(0)
    model = ctor()
    model.eval()
    other_edge_attr = torch.randn_like(graph["edge_attr"]) * 5 + 3
    with torch.no_grad():
        a = model(graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], graph["fp"])
        b = model(graph["x"], graph["edge_index"], other_edge_attr, graph["batch"], graph["fp"])
    assert not torch.allclose(a, b), f"{name} ignores edge_attr"


def test_edge_dim_of_zero_is_treated_as_absent():
    """A featurizer that produces no bond features must not create Linear(0, ...) layers."""
    assert models._edge_dim_or_none(0) is None
    assert models._edge_dim_or_none(None) is None
    assert models._edge_dim_or_none(11) == 11


# ------------------------------------------------------------------------------ MLPModel

def test_mlp_model_runs_on_fingerprints_alone():
    model = MLPModel(MLP[0], MLP[1], MLP[2], MLP[3], *PRED, N_TARGETS)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, MLP[0]))
    assert out.shape == (4, N_TARGETS)


def test_mlp_model_rejects_zero_output_width():
    """Regression: this used to build a Linear(0, ...) and fail later with an opaque
    shape error instead of the clear message every other model gives."""
    with pytest.raises(ValueError, match="cannot be 0"):
        MLPModel(MLP[0], MLP[1], MLP[2], 0, *PRED, N_TARGETS)


def test_prediction_head_hidden_layers_can_be_skipped():
    """predictor_n_layers has a minimum of 0 in the UI."""
    model = MLPModel(MLP[0], MLP[1], MLP[2], MLP[3], 32, 0, N_TARGETS)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(3, MLP[0])).shape == (3, N_TARGETS)


def test_multiple_targets():
    model = MLPModel(MLP[0], MLP[1], MLP[2], MLP[3], *PRED, 5)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(3, MLP[0])).shape == (3, 5)


# ------------------------------------------------------------------------------ training

def test_gradients_flow_through_both_branches(graph):
    model = build_custom("GATConv", 2)
    out = model(graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], graph["fp"])
    out.sum().backward()

    graph_encoder_grads = [p.grad for p in model.graph_encoder.parameters() if p.grad is not None]
    fingerprint_encoder_grads = [p.grad for p in model.fingerprint_encoder.parameters() if p.grad is not None]
    assert graph_encoder_grads and any(g.abs().sum() > 0 for g in graph_encoder_grads)
    assert fingerprint_encoder_grads and any(g.abs().sum() > 0 for g in fingerprint_encoder_grads)


@pytest.mark.parametrize("layer_name", ["GENConv", "TransformerConv"])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_single_atom_training_batch_uses_running_normalization(layer_name, n_layers, graph):
    """A final training batch can contain only a one-atom molecule (e.g. methane).
    BatchNorm must use its running statistics without corrupting them or disabling
    gradient computation for that sample.
    """
    model = build_custom(layer_name, n_layers)
    # First obtain representative running statistics from a normal training batch.
    model(graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], graph["fp"])
    norms = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    before = [{key: value.clone() for key, value in layer.state_dict().items()} for layer in norms]
    single = (
        graph["x"][:1], torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, EDGE_DIM)), torch.zeros(1, dtype=torch.long), graph["fp"][:1],
    )

    out = model(*single)
    assert out.shape == (1, N_TARGETS)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for norm in norms for parameter in norm.parameters())
    for norm, state in zip(norms, before):
        assert norm.training
        for key, value in norm.state_dict().items():
            torch.testing.assert_close(value, state[key])

    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(out, model(*single))


@pytest.mark.parametrize("name", ALL_CUSTOM_LAYERS + list(TOP_LEVEL))
def test_edge_feature_dispatch_uses_model_type(name):
    model = build_custom(name, 1) if name in ALL_CUSTOM_LAYERS else TOP_LEVEL[name][0]()
    assert models.model_requires_edge_attr(model) == (name in EDGE_ATTR_LAYERS or name in {"GAT", "AttentiveFP"})


@pytest.mark.parametrize("name", [*ALL_CUSTOM_LAYERS, *TOP_LEVEL, "MLP"])
def test_model_architecture_and_checkpoint_keys_use_encoder_names(name):
    if name == "MLP":
        model = MLPModel(*MLP, *PRED, N_TARGETS)
        expected = {"fingerprint_encoder", "prediction_head"}
    else:
        model = build_custom(name, 2) if name in ALL_CUSTOM_LAYERS else TOP_LEVEL[name][0]()
        expected = {"graph_encoder", "fingerprint_encoder", "prediction_head"}
    registered = dict(model.named_children())
    keys = {key.split(".", 1)[0] for key in model.state_dict()}
    assert expected <= registered.keys()
    assert expected <= keys
    assert {"gcn", "mlp", "predictor"}.isdisjoint(registered)
    assert {"gcn", "mlp", "predictor"}.isdisjoint(keys)
    for module_name in expected:
        assert f"({module_name}):" in repr(model)


@pytest.mark.parametrize("layer_name", ["GENConv", "TransformerConv"])
def test_single_node_normalization_keeps_legacy_checkpoints_and_batch_behavior(layer_name, graph, monkeypatch):
    # Old checkpoints use plain BatchNorm1d parameters and running statistics.
    with monkeypatch.context() as context:
        context.setattr(models, "_SingleNodeBatchNorm1d", torch.nn.BatchNorm1d)
        legacy = build_custom(layer_name, 2)
    current = build_custom(layer_name, 2)
    current.load_state_dict(legacy.state_dict(), strict=True)
    args = (graph["x"], graph["edge_index"], graph["edge_attr"], graph["batch"], graph["fp"])

    torch.testing.assert_close(current(*args), legacy(*args))
    for key, value in current.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key])
