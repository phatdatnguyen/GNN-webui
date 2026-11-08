import torch
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import GCN, GraphSAGE, GIN, GAT, EdgeCNN, AttentiveFP, GCNConv, SAGEConv, SGConv, ClusterGCNConv, GraphConv, ChebConv, LEConv, EGConv, MFConv, FeaStConv, TAGConv, ARMAConv, FiLMConv, PDNConv, GENConv, ResGatedGraphConv, GATConv, GATv2Conv, SuperGATConv, TransformerConv, GeneralConv, global_mean_pool

# Chebyshev filter size for ChebConv. It has no dedicated UI control, so it is fixed here.
CHEB_CONV_K = 3


def model_requires_edge_attr(model):
    """Select the forward signature from the actual model, independent of UI tabs."""
    return isinstance(model, (
        GATModel, AttentiveFPModel, PDNConvModel, GENConvModel,
        ResGatedGraphConvModel, GATConvModel, GATv2ConvModel,
        TransformerConvModel, GeneralConvModel,
    ))


def _edge_dim_or_none(edge_dim):
    # Layers that take an optional edge_dim expect None (not 0) when there are no edge features.
    return edge_dim if edge_dim else None


class _SingleNodeBatchNorm1d(BatchNorm1d):
    """Keep one-atom training batches valid using the existing running statistics."""

    def forward(self, x):
        if self.training and x.dim() == 2 and x.size(0) == 1:
            return torch.nn.functional.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps,
            )
        return super().forward(x)


def _gen_conv(n_inputs, n_hiddens, edge_dim):
    conv = GENConv(n_inputs, n_hiddens, edge_dim=_edge_dim_or_none(edge_dim))
    # GENConv also owns BatchNorm layers inside its MLP. Replacing them at
    # construction preserves the parameter names and existing checkpoint format.
    for i, layer in enumerate(conv.mlp):
        if isinstance(layer, BatchNorm1d):
            replacement = _SingleNodeBatchNorm1d(
                layer.num_features, eps=layer.eps, momentum=layer.momentum,
                affine=layer.affine, track_running_stats=layer.track_running_stats,
            )
            replacement.load_state_dict(layer.state_dict())
            conv.mlp[i] = replacement
    return conv


class MLPModel(torch.nn.Module):
    def __init__(
        self,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_mlp_outputs == 0:
            raise ValueError("The output size of the Fingerprint Encoder cannot be 0!")

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ])

        prediction_head_input_dim = n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, mol_features):
        h = mol_features

        if self.fingerprint_encoder:
            for linear in self.fingerprint_encoder:
                h = torch.relu(linear(h))

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GCNModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = GCN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            # BasicGNN.forward is (x, edge_index, edge_weight, edge_attr, batch, ...)
            h1 = self.graph_encoder(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        # Combine h1 and h2 if both exist, else use whichever is not None
        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GraphSAGEModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = GraphSAGE(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.graph_encoder(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
    
class GINModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = GIN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.graph_encoder(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
    
class GATModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        edge_dim: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        # edge_dim must be given to the underlying GATConv layers, otherwise edge_attr is silently discarded.
        self.graph_encoder = GAT(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout,
                       edge_dim=_edge_dim_or_none(edge_dim)) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.graph_encoder(x, edge_index, None, edge_attr, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class EdgeCNNModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = EdgeCNN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.graph_encoder(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class AttentiveFPModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        edge_dim: int,
        n_gcn_layers: int,
        num_timesteps: int,
        n_gcn_outputs: int,
        n_gcn_dropout: float,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = AttentiveFP(
            in_channels=n_gcn_inputs,
            hidden_channels=n_gcn_hiddens,
            edge_dim=edge_dim,
            num_layers=n_gcn_layers,
            num_timesteps=num_timesteps,
            out_channels=n_gcn_outputs,
            dropout=n_gcn_dropout
        ) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.graph_encoder(x, edge_index, edge_attr, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GCNConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            GCNConv(n_gcn_inputs, n_gcn_hiddens),
            *[GCNConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, GCNConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:  # Last layer is Linear
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
    
class SAGEConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            SAGEConv(n_gcn_inputs, n_gcn_hiddens),
            *[SAGEConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, SAGEConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class SGConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            SGConv(n_gcn_inputs, n_gcn_hiddens),
            *[SGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, SGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class ClusterGCNConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            ClusterGCNConv(n_gcn_inputs, n_gcn_hiddens),
            *[ClusterGCNConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, ClusterGCNConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GraphConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            GraphConv(n_gcn_inputs, n_gcn_hiddens),
            *[GraphConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, GraphConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
    
class ChebConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            ChebConv(n_gcn_inputs, n_gcn_hiddens, CHEB_CONV_K),
            *[ChebConv(n_gcn_hiddens, n_gcn_hiddens, CHEB_CONV_K) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, ChebConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class LEConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            LEConv(n_gcn_inputs, n_gcn_hiddens),
            *[LEConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None
        
        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, LEConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class EGConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            EGConv(n_gcn_inputs, n_gcn_hiddens),
            *[EGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, EGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class MFConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            MFConv(n_gcn_inputs, n_gcn_hiddens),
            *[MFConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, MFConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class FeaStConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            FeaStConv(n_gcn_inputs, n_gcn_hiddens),
            *[FeaStConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, FeaStConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class TAGConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            TAGConv(n_gcn_inputs, n_gcn_hiddens),
            *[TAGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, TAGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class ARMAConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            ARMAConv(n_gcn_inputs, n_gcn_hiddens),
            *[ARMAConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, ARMAConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class FiLMConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        self.graph_encoder = ModuleList([
            FiLMConv(n_gcn_inputs, n_gcn_hiddens),
            *[FiLMConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            h1 = x
            for i, layer in enumerate(self.graph_encoder):
                if isinstance(layer, FiLMConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class PDNConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        edge_dim: int,
        edge_n_hiddens: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(PDNConv(n_gcn_inputs, n_gcn_hiddens, edge_dim, edge_n_hiddens))
            self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(PDNConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim, edge_n_hiddens))
                self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GENConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(_gen_conv(n_gcn_inputs, n_gcn_hiddens, edge_dim))
            self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(_gen_conv(n_gcn_hiddens, n_gcn_hiddens, edge_dim))
                self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class ResGatedGraphConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(ResGatedGraphConv(n_gcn_inputs, n_gcn_hiddens, edge_dim=_edge_dim_or_none(edge_dim)))
            self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(ResGatedGraphConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim=_edge_dim_or_none(edge_dim)))
                self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GATConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_heads: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(GATConv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
            self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(GATConv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
                self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GATv2ConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_heads: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(GATv2Conv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
            self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(GATv2Conv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
                self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
    
class SuperGATConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_heads: int,
        n_gcn_outputs: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(SuperGATConv(n_gcn_inputs, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
            self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(SuperGATConv(n_gcn_hiddens, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
                self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
   
class TransformerConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_heads: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            first_out = n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs
            self.graph_encoder.append(TransformerConv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
            self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, first_out))
            self.graph_encoder.append(_SingleNodeBatchNorm1d(first_out))
            for i in range(1, n_gcn_layers):
                layer_out = n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs
                self.graph_encoder.append(TransformerConv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=_edge_dim_or_none(edge_dim)))
                self.graph_encoder.append(Linear(n_gcn_hiddens*n_gcn_heads, layer_out))
                self.graph_encoder.append(_SingleNodeBatchNorm1d(layer_out))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 3 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                elif i % 3 == 1:
                    h1 = torch.relu(layer(h1))
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)

class GeneralConvModel(torch.nn.Module):
    def __init__(
        self,
        n_gcn_inputs: int,
        n_gcn_hiddens: int,
        n_gcn_layers: int,
        n_gcn_heads: int,
        n_gcn_outputs: int,
        edge_dim: int,
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int,
        n_outputs: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of the Graph Encoder and Fingerprint Encoder cannot be 0!")

        if n_gcn_outputs > 0:
            self.graph_encoder = ModuleList()
            self.graph_encoder.append(GeneralConv(n_gcn_inputs, n_gcn_hiddens, in_edge_channels=_edge_dim_or_none(edge_dim), attention=True, heads=n_gcn_heads))
            self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if n_gcn_layers > 1 else n_gcn_outputs))
            for i in range(1, n_gcn_layers):
                self.graph_encoder.append(GeneralConv(n_gcn_hiddens, n_gcn_hiddens, in_edge_channels=_edge_dim_or_none(edge_dim), attention=True, heads=n_gcn_heads))
                self.graph_encoder.append(Linear(n_gcn_hiddens, n_gcn_hiddens if i != n_gcn_layers - 1 else n_gcn_outputs))
        else:
            self.graph_encoder = None

        self.fingerprint_encoder = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        prediction_head_input_dim = n_gcn_outputs + n_mlp_outputs
        self.prediction_head = ModuleList([
            Linear(prediction_head_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, n_outputs) if n_predictor_layers > 0 else Linear(prediction_head_input_dim, n_outputs)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.graph_encoder:
            for i, layer in enumerate(self.graph_encoder):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.fingerprint_encoder:
            h2 = mol_features
            for linear in self.fingerprint_encoder:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.prediction_head:
            for linear in self.prediction_head:
                h = torch.relu(linear(h))

        return self.out(h)
