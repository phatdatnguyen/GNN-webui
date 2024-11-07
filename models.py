import torch
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import GCNConv, SAGEConv, CuGraphSAGEConv, SGConv, ClusterGCNConv, GraphConv, ChebConv, LEConv, EGConv, MFConv, FeaStConv, TAGConv, ARMAConv, FiLMConv, PDNConv, GENConv, ResGatedGraphConv, GATConv, GATv2Conv, SuperGATConv, TransformerConv, GeneralConv, global_mean_pool

class GCNModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GCNModel, self).__init__()

        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GCNConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GCNConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
           
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
    
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GraphSAGEModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(SAGEConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(SAGEConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
    
class CuGraphSAGEModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(CuGraphSAGEModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(CuGraphSAGEConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(CuGraphSAGEConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class SGConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(SGConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(SGConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(SGConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class ClusterGCNModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(ClusterGCNModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(ClusterGCNConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(ClusterGCNConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class GraphConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GraphConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GraphConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GraphConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
    
class ChebConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(ChebConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(ChebConv(n_gcn_inputs, n_gcn_hiddens, 2))
            for i in range(1, n_gcn_layers):
                self.gcn.append(ChebConv(n_gcn_hiddens, n_gcn_hiddens, 2))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class LEConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(LEConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(LEConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(LEConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class EGConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(EGConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(EGConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(EGConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class MFConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(MFConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(MFConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(MFConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        return self.out(h)

class FeaStConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(FeaStConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(FeaStConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(FeaStConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class TAGConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(TAGConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(TAGConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(TAGConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class ARMAConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(ARMAConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(ARMAConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(ARMAConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class FiLMConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(FiLMConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")

        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(FiLMConv(n_gcn_inputs, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(FiLMConv(n_gcn_hiddens, n_gcn_hiddens))
            
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i < len(self.gcn) - 1:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1))  
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class PDNConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, edge_dim, edge_n_hiddens,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(PDNConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(PDNConv(n_gcn_inputs, n_gcn_hiddens, edge_dim, edge_n_hiddens))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(PDNConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim, edge_n_hiddens))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class GENConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, edge_dim,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GENConvModel, self).__init__()

        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GENConv(n_gcn_inputs, n_gcn_hiddens, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GENConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class ResGatedGraphConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_outputs, edge_dim,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(ResGatedGraphConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(ResGatedGraphConv(n_gcn_inputs, n_gcn_hiddens, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(ResGatedGraphConv(n_gcn_hiddens, n_gcn_hiddens, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class GATModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_heads, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GATModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GATConv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GATConv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class GATv2Model(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_heads, n_gcn_outputs, edge_dim,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GATv2Model, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GATv2Conv(n_gcn_inputs, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GATv2Conv(n_gcn_hiddens, n_gcn_hiddens, n_gcn_heads, edge_dim=edge_dim))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
    
class SuperGATModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_heads, n_gcn_outputs,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(SuperGATModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(SuperGATConv(n_gcn_inputs, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
            self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(SuperGATConv(n_gcn_hiddens, n_gcn_hiddens, heads=n_gcn_heads, is_undirected=True))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens*n_gcn_heads, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
   
class TransformerConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_heads, n_gcn_outputs, edge_dim,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(TransformerConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
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
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 3 == 0: # GCN layer
                    h1 = layer(h1, edge_index, edge_attr)
                elif i % 3 == 1: # Linear layer
                    h1 = torch.relu(layer(h1)) 
                else: # BatchNorm1d layer
                    h1 = layer(h1)
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)

class GeneralConvModel(torch.nn.Module):
    def __init__(self, n_gcn_inputs, n_gcn_hiddens, n_gcn_layers, n_gcn_heads, n_gcn_outputs, edge_dim,
                 n_mlp_inputs, n_mlp_hiddens, n_mlp_layers, n_mlp_outputs,
                 n_predictor_hiddens, n_predictor_layers):
        super(GeneralConvModel, self).__init__()
        
        # Check output size
        if n_gcn_outputs == 0 and n_mlp_outputs == 0:
            raise Exception("The total output size of GCN and MLP modules cannot be 0!")
        
        # GCN layers
        self.gcn = torch.nn.ModuleList()
        if n_gcn_outputs > 0:
            self.gcn.append(GeneralConv(n_gcn_inputs, n_gcn_hiddens, in_edge_channels=edge_dim, attention=True, heads=n_gcn_heads))
            self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
            for i in range(1, n_gcn_layers):
                self.gcn.append(GeneralConv(n_gcn_hiddens, n_gcn_hiddens, in_edge_channels=edge_dim, attention=True, heads=n_gcn_heads))
                if i != n_gcn_layers - 1:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_hiddens))
                else:
                    self.gcn.append(Linear(n_gcn_hiddens, n_gcn_outputs))
        
        # MLP layers
        self.mlp = ModuleList()
        if n_mlp_outputs > 0:
            self.mlp.append(Linear(n_mlp_inputs, n_mlp_hiddens))
            for i in range(1, n_mlp_layers):
                self.mlp.append(Linear(n_mlp_hiddens, n_mlp_hiddens))
            self.mlp.append(Linear(n_mlp_hiddens, n_mlp_outputs))
        
        # Predictor layers
        self.predictor = ModuleList()
        self.predictor.append(Linear(n_gcn_outputs + n_mlp_outputs, n_predictor_hiddens))
        for i in range(1, n_predictor_layers):
            self.predictor.append(Linear(n_predictor_hiddens, n_predictor_hiddens))
        
        self.out = Linear(n_predictor_hiddens, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index, mol_features):
        if len(self.gcn) > 0:
            for i, layer in enumerate(self.gcn):
                if i == 0:
                    h1 = layer(x, edge_index, edge_attr)
                elif i % 2 == 0:
                    h1 = layer(h1, edge_index, edge_attr)
                else:
                    h1 = torch.relu(layer(h1)) 
            h1 = global_mean_pool(h1, batch_index)
        else:
            h1 = None

        if len(self.mlp) > 0:
            h2 = mol_features
            for i, linear in enumerate(self.mlp):
                h2 = torch.relu(linear(h2))
        else:
            h2 = None

        if h1 != None and h2 != None:
            h = torch.cat((h1, h2), dim=1)
        elif h1 != None:
            h = h1
        elif h2 != None:
            h = h2
        
        for i, linear in enumerate(self.predictor):
            h = torch.relu(linear(h))
        
        return self.out(h)
