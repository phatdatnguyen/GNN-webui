import torch
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import GCN, GraphSAGE, GIN, GAT, EdgeCNN, AttentiveFP, GCNConv, SAGEConv, SGConv, ClusterGCNConv, GraphConv, ChebConv, LEConv, EGConv, MFConv, FeaStConv, TAGConv, ARMAConv, FiLMConv, PDNConv, GENConv, ResGatedGraphConv, GATConv, GATv2Conv, SuperGATConv, TransformerConv, GeneralConv, global_mean_pool

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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = GCN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        # Combine h1 and h2 if both exist, else use whichever is not None
        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = GraphSAGE(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = GIN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
                h = torch.relu(linear(h))

        return self.out(h)
    
class GATModel(torch.nn.Module):
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = GAT(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, None, edge_attr, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = EdgeCNN(n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, n_gcn_dropout) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, None, None, batch_index)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = AttentiveFP(
            in_channels=n_gcn_inputs,
            hidden_channels=n_gcn_hiddens,
            edge_dim=edge_dim,
            num_layers=n_gcn_layers,
            num_timesteps=num_timesteps,
            out_channels=n_gcn_outputs,
            dropout=n_gcn_dropout
        ) if n_gcn_outputs > 0 else None
        self.n_gcn_outputs = n_gcn_outputs

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.n_gcn_outputs > 0:
            h1 = self.gcn(x, edge_index, edge_attr, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            GCNConv(n_gcn_inputs, n_gcn_hiddens),
            *[GCNConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, GCNConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:  # Last layer is Linear
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            SAGEConv(n_gcn_inputs, n_gcn_hiddens),
            *[SAGEConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, SAGEConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            SGConv(n_gcn_inputs, n_gcn_hiddens),
            *[SGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, SGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            ClusterGCNConv(n_gcn_inputs, n_gcn_hiddens),
            *[ClusterGCNConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, ClusterGCNConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            GraphConv(n_gcn_inputs, n_gcn_hiddens),
            *[GraphConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, GraphConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            ChebConv(n_gcn_inputs, n_gcn_hiddens),
            *[ChebConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, ChebConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            LEConv(n_gcn_inputs, n_gcn_hiddens),
            *[LEConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None
        
        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, LEConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            EGConv(n_gcn_inputs, n_gcn_hiddens),
            *[EGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, EGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            MFConv(n_gcn_inputs, n_gcn_hiddens),
            *[MFConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, MFConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            FeaStConv(n_gcn_inputs, n_gcn_hiddens),
            *[FeaStConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, FeaStConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            TAGConv(n_gcn_inputs, n_gcn_hiddens),
            *[TAGConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, TAGConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            ARMAConv(n_gcn_inputs, n_gcn_hiddens),
            *[ARMAConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, ARMAConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        self.gcn = ModuleList([
            FiLMConv(n_gcn_inputs, n_gcn_hiddens),
            *[FiLMConv(n_gcn_hiddens, n_gcn_hiddens) for _ in range(1, n_gcn_layers)],
            Linear(n_gcn_hiddens, n_gcn_outputs)
        ]) if n_gcn_outputs > 0 else None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            h1 = x
            for i, layer in enumerate(self.gcn):
                if isinstance(layer, FiLMConv):
                    h1 = layer(h1, edge_index)
                    h1 = torch.relu(h1)
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(PDNConv(n_gcn_inputs, n_gcn_hiddens, edge_dim, edge_n_hiddens))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(PDNConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim, edge_n_hiddens))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(GENConv(n_gcn_inputs, n_gcn_hiddens, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GENConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(ResGatedGraphConv(n_gcn_inputs, n_gcn_hiddens, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(ResGatedGraphConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_mlp_inputs: int,
        n_mlp_hiddens: int,
        n_mlp_layers: int,
        n_mlp_outputs: int,
        n_predictor_hiddens: int,
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(GATConv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GATConv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(GATv2Conv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GATv2Conv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(SuperGATConv(n_gcn_inputs, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(SuperGATConv(n_gcn_hiddens, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(TransformerConv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            self.gcn.append(BatchNorm1d(n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(TransformerConv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                    self.gcn.append(BatchNorm1d(n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
                    self.gcn.append(BatchNorm1d(n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 3 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                elif i % 3 == 1:
                    h1 = torch.relu(layer(h1))
                else:
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
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
        n_predictor_layers: int
    ):
        super().__init__()

        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise ValueError("The total output size of GCN and MLP modules cannot be 0!")

        if n_gcn_outputs > 0:
            self.gcn = ModuleList()
            self.gcn.append(GeneralConv(n_gcn_inputs, n_gcn_hiddens, in_edge_channels=edge_dim, attention=True, heads=n_gcn_heads))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GeneralConv(n_gcn_hiddens, n_gcn_hiddens, in_edge_channels=edge_dim, attention=True, heads=n_gcn_heads))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        else:
            self.gcn = None

        self.mlp = ModuleList([
            Linear(n_mlp_inputs, n_mlp_hiddens),
            *[Linear(n_mlp_hiddens, n_mlp_hiddens) for _ in range(1, n_mlp_layers)],
            Linear(n_mlp_hiddens, n_mlp_outputs)
        ]) if n_mlp_outputs > 0 else None

        predictor_input_dim = n_gcn_outputs + n_mlp_outputs
        self.predictor = ModuleList([
            Linear(predictor_input_dim, n_predictor_hiddens),
            *[Linear(n_predictor_hiddens, n_predictor_hiddens) for _ in range(1, n_predictor_layers)]
        ]) if n_predictor_layers > 0 else None

        self.out = Linear(n_predictor_hiddens, 1) if n_predictor_layers > 0 else Linear(predictor_input_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        h1 = None
        h2 = None
        if self.gcn:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1))
            h1 = global_mean_pool(h1, batch_index)

        if self.mlp:
            h2 = mol_features
            for linear in self.mlp:
                h2 = torch.relu(linear(h2))

        if h1 is not None and h2 is not None:
            h = torch.cat((h1, h2), dim=1)
        else:
            h = h1 if h1 is not None else h2

        if self.predictor:
            for linear in self.predictor:
                h = torch.relu(linear(h))

        return self.out(h)
