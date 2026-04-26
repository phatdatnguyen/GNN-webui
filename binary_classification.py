import math
import os
import time
import pandas as pd
import gradio as gr
import torch
from torch_geometric.nn import summary
from torch_geometric.loader import DataLoader
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from matplotlib import pyplot as plt
from datasets import MolecularDataset
from models import *
from utils import *

def get_working_directories():
    base_path = "./Data/"
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def get_working_directory_contents(working_directory_path):
    contents = [f for f in os.listdir(working_directory_path)]
    return contents

def on_open_working_directory(working_directory_name):
    if working_directory_name is None or working_directory_name.strip() == "":
        gr.Warning("Please specify a working directory.")
        return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), None, None
    
    working_directory_path = os.path.join("./Data/", working_directory_name)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_working_directory_contents(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory_name), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), working_directory_path, files

def on_content_list_change(working_directory_path, dataset_file_name, graph_directory, molecular_fingerprint_directory, checkpoint_file_name):
    contents = get_working_directory_contents(working_directory_path)
    # Update the file dataframe
    contents_info = []
    for content in contents:
        content_path = os.path.join(working_directory_path, content)
        if os.path.isdir(content_path):
            file_type = "Directory"
        elif content.endswith('.ckpt'):
            file_type = "Checkpoint file"
        elif content.endswith('.csv'):
            file_type = "Data file"
        else:
            file_type = "Other file"
        modified_time = time.ctime(os.path.getmtime(content_path))
        contents_info.append([content, file_type, modified_time])
        contents_info.sort(key=lambda x: x[2].lower(), reverse=True) # Sort by modified time descending
    contents_df = pd.DataFrame(contents_info, columns=["File", "Type", "Modified"])

    # Filter files
    data_files = [f for f in contents if f.endswith('.csv')]
    directories = [f for f in contents if os.path.isdir(os.path.join(working_directory_path, f))]
    checkpoint_files = [f for f in contents if f.endswith('.ckpt')]

    # Update the file dropdown choices and value
    if f"{dataset_file_name}.csv" in data_files:
        dataset_file_name_value = f"{dataset_file_name}.csv"
    else:
        dataset_file_name_value = data_files[0] if data_files else None

    if graph_directory in directories:
        graph_directory_value = graph_directory
    else:
        graph_directory_value = directories[0] if directories else None

    if molecular_fingerprint_directory in directories:
        molecular_fingerprint_directory_value = molecular_fingerprint_directory
    else:
        molecular_fingerprint_directory_value = directories[0] if directories else None

    if checkpoint_file_name in checkpoint_files:
        checkpoint_file_name_value = checkpoint_file_name
    else:
        checkpoint_file_name_value = checkpoint_files[0] if checkpoint_files else None

    if f"{dataset_file_name}.csv" in data_files:
        prediction_dataset_file_name_value = f"{dataset_file_name}.csv"
    else:
        prediction_dataset_file_name_value = data_files[0] if data_files else None

    if graph_directory in directories:
        prediction_graph_directory_value = graph_directory
    else:
        prediction_graph_directory_value = directories[0] if directories else None

    if molecular_fingerprint_directory in directories:
        prediction_molecular_fingerprint_directory_value = molecular_fingerprint_directory
    else:
        prediction_molecular_fingerprint_directory_value = directories[0] if directories else None

    return contents_df, \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=directories, value=graph_directory_value), \
           gr.update(choices=directories, value=molecular_fingerprint_directory_value), \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=checkpoint_files, value=checkpoint_file_name_value), \
           gr.update(choices=data_files, value=prediction_dataset_file_name_value), \
           gr.update(choices=directories, value=prediction_graph_directory_value), \
           gr.update(choices=directories, value=prediction_molecular_fingerprint_directory_value)

def get_device_dropdown():
    if torch.cuda.is_available():
        return gr.Dropdown(label="Device", choices=[("CUDA", "cuda"), ("CPU", "cpu")], value="cuda")
    else:
        return gr.Dropdown(label="Device", choices=[("CPU", "cpu")], value="cpu", interactive=False)

def on_datatype_change(datatype):
    if datatype == "float16":
        return torch.float16
    elif datatype == "float32":
        return torch.float32
    elif datatype == "float64":
        return torch.float64
    else: # datatype == "bfloat16"
        return torch.bfloat16

def on_upload_dataset(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns):
    if dataset_file_path is None or dataset_file_path.strip() == "":
        status = "Please select a dataset file."
        return f"<span style='color: red;'>{status}</span>", None, get_working_directory_contents(working_directory_path)
    
    status, df = process_dataset_file(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns)
    if df is not None:
        return f"<span style='color: green;'>{status}</span>", df.head(10), get_working_directory_contents(working_directory_path)
    else:
        return f"<span style='color: red;'>{status}</span>", None, get_working_directory_contents(working_directory_path)

def on_extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress=gr.Progress()):
    status = extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress)
    return status, get_working_directory_contents(working_directory_path)

def on_molecular_fingerprint_change(molecular_fingerpint_name):
    return gr.update(visible=(molecular_fingerpint_name == 'Morgan fingerprint'))

def on_load_data_option_change(load_data_option):
    load_graphs = (load_data_option == 'Graphs' or load_data_option == 'Both')
    load_fingerprints = (load_data_option == 'Molecular fingerprints' or load_data_option == 'Both')
    return gr.update(visible=load_graphs), gr.update(visible=load_fingerprints), gr.update(visible=load_fingerprints), gr.update(visible=load_fingerprints), gr.update(visible=load_fingerprints), gr.update(visible=load_graphs), gr.update(visible=load_fingerprints), gr.update(visible=load_graphs), gr.update(visible=load_fingerprints)

def on_extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress=gr.Progress()):
    status = extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress)
    return status, get_working_directory_contents(working_directory_path)

def on_dimensionality_reduction_change(dimensionality_reduction_enabled):
    return gr.update(visible=dimensionality_reduction_enabled), gr.update(visible=dimensionality_reduction_enabled)

def on_process_data(working_directory_path, dataset_file_name, load_data, graph_directory, molecular_fingerprint_directory, datatype, dimensionality_reduction, variance_threshold, pca_num_components, test_ratio, val_ratio, batch_size, random_seed):
    try:
        # Validate inputs
        if load_data in ('Graphs', 'Both') and (graph_directory is None or graph_directory.strip() == ""):
            raise ValueError("Please select a graph directory.")
        if load_data in ('Molecular fingerprints', 'Both') and (molecular_fingerprint_directory is None or molecular_fingerprint_directory.strip() == ""):
            raise ValueError("Please select a molecular fingerprint directory.")

        # Validate output columns
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        target_columns_list = df.drop(columns=['SMILES']).columns.tolist()
        if len(target_columns_list) == 0:
            raise ValueError("No target columns found in the dataset. Please ensure your dataset has at least one target column in addition to the SMILES column.")
        for col in target_columns_list:
            # Drop NaN values and get unique values
            unique_values = df[col].dropna().unique()
            num_unique = len(unique_values)
            
            # Check if there are exactly 2 unique values (0 and 1)
            is_binary = (num_unique == 2 and set(unique_values) == {0, 1})
            if not is_binary:
                raise ValueError(f"Target column '{col}' is not binary. Please ensure all target columns contain only two unique values, 0 and 1.")

        # Set up dataset and dataloaders
        graph_directory_path = os.path.join(working_directory_path, graph_directory)
        molecular_fingerprint_directory_path = os.path.join(working_directory_path, molecular_fingerprint_directory)

        dataset = MolecularDataset(dataset_file_path, load_data, graph_directory_path, molecular_fingerprint_directory_path, datatype)
        
        total_len = len(dataset)
        test_size = int(total_len * test_ratio)
        val_size = int(total_len * val_ratio)
        train_size = total_len - test_size - val_size
        if train_size <= 0 or val_size < 0 or test_size < 0:
            raise ValueError("Invalid split sizes. Please adjust test/val sizes.")
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Set up feature transformers
        if dimensionality_reduction:
            variance_threshold = VarianceThreshold(variance_threshold)
            pca = PCA(n_components=pca_num_components)
            fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, "molecular_fingerprints.csv"))
            fingerprints = fingerprint_df.drop(columns=['SMILES']).values
            fingerprints_variance_thresholded = variance_threshold.fit_transform(fingerprints)
            pca.fit(fingerprints_variance_thresholded)
        else:
            variance_threshold = None
            pca = None

        target_column_names = dataset.target_column_names

        status = f"Data processed: {total_len} molecules."
        return f"<span style='color: green;'>{status}</span>", gr.update(interactive=True), gr.update(choices=target_column_names, value=target_column_names[0]), dataset, train_dataloader, val_dataloader, test_dataloader, variance_threshold, pca
    except Exception as exc:
        status = f"Error processing data: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, None, None, None, None, None, None

def on_gnn_model_tab_selected(evt: gr.SelectData):
    return evt.value

def on_create_model(dataset, gnn_model_tab, device, datatype, random_seed, variance_threshold, pca,
                    gcn_n_hiddens, gcn_num_layers, gcn_dropout, gcn_n_outputs,
                    graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_dropout, graph_sage_n_outputs,
                    gin_n_hiddens, gin_num_layers, gin_dropout, gin_n_outputs,
                    gat_n_hiddens, gat_num_layers, gat_dropout, gat_n_outputs,
                    edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_dropout, edgecnn_n_outputs,
                    attentivefp_n_hiddens, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_dropout, attentivefp_n_outputs,
                    customgcn_convolutional_layer_name, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                    mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                    predictor_n_hiddens, predictor_n_layers):
    
    # Initialize model parameters based on dataset and user selections
    if dataset.load_data == 'Graphs':
        gcn_n_inputs = dataset.num_node_features
        edge_dim = dataset.num_edge_features
        mlp_n_inputs = 0
        mlp_n_outputs = 0
    elif dataset.load_data == 'Molecular fingerprints':
        gcn_n_inputs = 0
        edge_dim = 0
        gcn_n_outputs = 0
        graph_sage_n_outputs = 0
        gin_n_outputs = 0
        gat_n_outputs = 0
        edgecnn_n_outputs = 0
        attentivefp_n_outputs = 0
        customgcn_n_outputs = 0
        mlp_n_inputs = dataset.num_fingerprint_bits
    else: # Both
        gcn_n_inputs = dataset.num_node_features
        edge_dim = dataset.num_edge_features
        mlp_n_inputs = dataset.num_fingerprint_bits
    n_outputs = dataset.num_outputs

    if variance_threshold is not None and pca is not None:
        mlp_n_inputs = pca.n_components_

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)

    model = None
    train_losses = []
    val_losses = []
    trained_epochs = 0

    try:
        # Create the model based on the selected GNN architecture
        if dataset.load_data == 'Molecular fingerprints':
            model = MLPModel(mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "GCN":
            model = GCNModel(gcn_n_inputs, gcn_n_hiddens, gcn_num_layers, gcn_n_outputs, gcn_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "GraphSAGE":
            model = GraphSAGEModel(gcn_n_inputs, graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_n_outputs, graph_sage_dropout,
                                   mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                   predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "GIN":
            model = GINModel(gcn_n_inputs, gin_n_hiddens, gin_num_layers, gin_n_outputs, gin_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "GAT":
            model = GATModel(gcn_n_inputs, gat_n_hiddens, gat_num_layers, gat_n_outputs, gat_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "EdgeCNN":
            model = EdgeCNNModel(gcn_n_inputs, edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_n_outputs, edgecnn_dropout,
                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                 predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        elif gnn_model_tab == "AttentiveFP":
            model = AttentiveFPModel(gcn_n_inputs, attentivefp_n_hiddens, edge_dim, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_n_outputs, attentivefp_dropout,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        else: # Custom
            if customgcn_convolutional_layer_name == "GCNConv":
                model = GCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "SAGEConv":
                model = SAGEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "SGConv":
                model = SGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "ClusterGCNConv":
                model = ClusterGCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                            predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "GraphConv":
                model = GraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                       mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                       predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "LEConv":
                model = LEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "EGConv":
                model = EGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "MFConv":
                model = MFConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "TAGConv":
                model = TAGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "ARMAConv":
                model = ARMAConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "FiLMConv":
                model = FiLMConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "PDNConv":
                model = PDNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim, gcn_n_hiddens,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "GENConv":
                model = GENConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "ResGatedGraphConv":
                model = ResGatedGraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "GATConv":
                model = GATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "GATv2Conv":
                model = GATv2ConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                       mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                       predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "SuperGATConv":
                model = SuperGATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                          mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                          predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            elif customgcn_convolutional_layer_name == "TransformerConv":
                model = TransformerConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                             predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
            else: # customgcn_convolutional_layer_name == "GeneralConv":
                model = GeneralConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                         mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                         predictor_n_hiddens, predictor_n_layers, n_outputs).to(device=device, dtype=datatype)
        
        # Reset training state
        train_losses = []
        val_losses = []
        trained_epochs = 0
        
        # Prepare a sample for model summary
        data = dataset[0]
        
        if dataset.load_data == 'Graphs':
            x = data.x.to(device=device, dtype=datatype)
            edge_index = data.edge_index.long().to(device=device)
            edge_attr = data.edge_attr.to(device=device, dtype=datatype)
            fingerprint = None
        elif dataset.load_data == 'Molecular fingerprints':
            x = None
            edge_index = None
            edge_attr = None
            fingerprint = data.fp
        else: # Both
            x = data.x.to(device=device, dtype=datatype)
            edge_index = data.edge_index.long().to(device=device)
            edge_attr = data.edge_attr.to(device=device, dtype=datatype)
            fingerprint = data.fp
        
        batch_index = torch.tensor([0], device=device, dtype=torch.int32)
        
        if fingerprint is not None:
            if variance_threshold is not None and pca is not None:
                fingerprint_variance_thresholded = variance_threshold.transform(fingerprint.reshape(1, -1))
                fingerprint_pca = pca.transform(fingerprint_variance_thresholded)
                fingerprint_tensor = torch.tensor(fingerprint_pca).to(device=device, dtype=datatype)
            else:
                fingerprint_tensor = torch.tensor(fingerprint.reshape(1, -1)).to(device=device, dtype=datatype)
        else:
            fingerprint_tensor = None

        # Model summary
        need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in ["PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
        if dataset.load_data == 'Molecular fingerprints':
            model_summary = summary(model, fingerprint_tensor)
        elif need_edge_attr:
            model_summary = summary(model, x, edge_index, edge_attr, batch_index, fingerprint_tensor)
        else:
            model_summary = summary(model, x, edge_index, batch_index, fingerprint_tensor)
        
        status = "Model created successfully."
        return f"<span style='color: green;'>{status}</span>", model_summary, \
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), \
            model, train_losses, val_losses, trained_epochs
    except Exception as exc:
        status = f"Error creating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, \
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), \
            model, train_losses, val_losses, trained_epochs

def on_save_checkpoint(working_directory_path, checkpoint_name, model, trained_epochs):
    try:
        checkpoint = {
            'state_dict': model.state_dict(),
            'trained_epochs': trained_epochs
        }
        checkpoint_path = os.path.join(working_directory_path, f"{checkpoint_name}_{trained_epochs}.ckpt")
        torch.save(checkpoint, checkpoint_path)
        status = f"Model checkpoint saved: {checkpoint_name}_{trained_epochs}.ckpt"
        return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error saving checkpoint: {exc}"
        return f"<span style='color: red;'>{status}</span>", get_working_directory_contents(working_directory_path)

def on_load_checkpoint(working_directory_path, checkpoint_file_name, model):
    try:
        checkpoint_file_path = os.path.join(working_directory_path, checkpoint_file_name)
        checkpoint = torch.load(checkpoint_file_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        trained_epochs = checkpoint.get('trained_epochs', 0)
        train_losses = []
        val_losses = []

        status = f"Checkpoint loaded: {checkpoint_file_name}."
        return f"<span style='color: green;'>{status}</span>", model, train_losses, val_losses, trained_epochs
    except Exception as exc:
        status = f"Error loading checkpoint: {exc}"
        return f"<span style='color: red;'>{status}</span>", model, None, None, None

loss_fn = torch.nn.BCEWithLogitsLoss()
def on_create_optimizer(model, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider):
    optimizers = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD
    }
    opt_class = optimizers.get(optimizer_dropdown)

    optimizer = opt_class(model.parameters(), lr=learning_rate_slider)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay_slider)

    status = "Optimizer created."
    return f"<span style='color: green;'>{status}</span>", optimizer, scheduler, gr.update(interactive=True)

# Define the train function
def train(model, dataset, dataloader, device, datatype, optimizer, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.train()
    total_loss = 0.0
    num_batches = 0
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    
    for batch in dataloader:
        # Get input data
        if dataset.load_data == 'Graphs' or dataset.load_data == 'Both':
            x = batch.x.to(device=device, dtype=datatype)
            edge_index = batch.edge_index.long().to(device=device)
            edge_attr = batch.edge_attr.to(device=device, dtype=datatype)
            batch_index = batch.batch.to(device=device)
        else:
            x = None
            edge_index = None
            edge_attr = None
            batch_index = None
        if dataset.load_data == 'Molecular fingerprints' or dataset.load_data == 'Both':
            fingerprints = batch.fp
            if variance_threshold is not None and pca is not None:
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints)
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None

        # Get target labels
        y = batch.y.to(device=device, dtype=datatype)
        
        # Forward pass
        optimizer.zero_grad()
        if dataset.load_data == 'Molecular fingerprints':
            output = model(fingerprints_tensor)
        elif need_edge_attr:
            output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
        else:
            output = model(x, edge_index, batch_index, fingerprints_tensor)

        # Compute loss and backpropagate
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += torch.sqrt(loss).item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

# Define the validation function
@torch.no_grad()
def validation(model, dataset, dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    
    for batch in dataloader:
        # Get input data
        if dataset.load_data == 'Graphs' or dataset.load_data == 'Both':
            x = batch.x.to(device=device, dtype=datatype)
            edge_index = batch.edge_index.long().to(device=device)
            edge_attr = batch.edge_attr.to(device=device, dtype=datatype)
            batch_index = batch.batch.to(device=device)
        else:
            x = None
            edge_index = None
            edge_attr = None
            batch_index = None
        if dataset.load_data == 'Molecular fingerprints' or dataset.load_data == 'Both':
            fingerprints = batch.fp
            if variance_threshold is not None and pca is not None:
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints)
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None
        
        # Get target labels
        y = batch.y.to(device=device, dtype=datatype)
        
        with torch.no_grad():
            # Forward pass
            if dataset.load_data == 'Molecular fingerprints':
                output = model(fingerprints_tensor)
            elif need_edge_attr:
                output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
            else:
                output = model(x, edge_index, batch_index, fingerprints_tensor)

            # Compute loss
            loss = loss_fn(output, y)
            total_loss += torch.sqrt(loss).item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

def on_train(model, dataset, train_dataloader, val_dataloader, device, datatype, optimizer, scheduler, epochs, trained_epochs, train_losses, val_losses, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca, progress=gr.Progress()):
    # Training loop
    for _ in progress.tqdm(range(trained_epochs, epochs + trained_epochs), total=epochs, desc="Training"):
        train_loss = train(model, dataset, train_dataloader, device, datatype, optimizer, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
        train_losses.append(train_loss)
        val_loss = validation(model, dataset, val_dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
        val_losses.append(val_loss)
        scheduler.step()
        trained_epochs += 1

    status = f"Training completed for {trained_epochs} epochs."

    # Plot the training and validation loss
    figure = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('sqrt(BCE with logits loss)')
    plt.legend()
    plt.grid(True)
    plt.close()

    export_losses_button = gr.Button(value="Export", interactive=True)

    return f"<span style='color: green;'>{status}</span>", figure, export_losses_button, trained_epochs, train_losses, val_losses

def on_export_losses(working_directory_path, file_name, train_losses, val_losses):
    df = pd.DataFrame()
    df['train_losses'] = train_losses
    df['val_losses'] = val_losses
    file_path = os.path.join(working_directory_path, f"{file_name}.csv")
    df.to_csv(file_path)
    status = f"Losses exported to {file_name}.csv."
    return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)

# Define the test function
@torch.no_grad()
def test(model, dataset, dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    y_test_list = []
    logits_list = []
    smiles_list = []
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    
    for batch in dataloader:
        # Get input data
        if dataset.load_data == 'Graphs' or dataset.load_data == 'Both':
            x = batch.x.to(device=device, dtype=datatype)
            edge_index = batch.edge_index.long().to(device=device)
            edge_attr = batch.edge_attr.to(device=device, dtype=datatype)
            batch_index = batch.batch.to(device=device)
        else:
            x = None
            edge_index = None
            edge_attr = None
            batch_index = None
        if dataset.load_data == 'Molecular fingerprints' or dataset.load_data == 'Both':
            fingerprints = batch.fp
            if variance_threshold is not None and pca is not None:
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints)
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None

        # Get target labels
        y = batch.y.to(device=device, dtype=datatype)
        smiles = batch.smiles

        with torch.no_grad():
            # Forward pass
            if dataset.load_data == 'Molecular fingerprints':
                output = model(fingerprints_tensor)
            elif need_edge_attr:
                output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
            else:
                output = model(x, edge_index, batch_index, fingerprints_tensor)

        smiles_list.extend(smiles)
        y_test_list.append(y.cpu().detach())
        logits_list.append(output.cpu().detach())

    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.tensor([])
    logits = torch.cat(logits_list, dim=0) if logits_list else torch.tensor([])
    return y_test, logits, smiles_list

def on_target_column_change(eval_df, target_column):
    if eval_df is None or eval_df.empty:
        return None, None, None
    
    # Evaluate the model using different metrics
    y_test_target = eval_df[target_column].to_numpy()
    y_pred_target = eval_df[f"{target_column} (Prediction)"].to_numpy()
    y_prob_target = eval_df[f"{target_column} (Probability)"].to_numpy()
    cm = confusion_matrix(y_test_target, y_pred_target)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_figure, ax = plt.subplots(figsize=(6, 6))
    display.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title('Confusion Matrix')

    accuracy = round(accuracy_score(y_test_target, y_pred_target), 4)
    precision = round(precision_score(y_test_target, y_pred_target), 4)
    recall = round(recall_score(y_test_target, y_pred_target), 4)
    f1 = round(f1_score(y_test_target, y_pred_target), 4)
    TN = cm[0, 0] if cm.shape[0] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    roc_auc = roc_auc_score(y_test_target, y_prob_target)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 score', 'ROC-AUC score'],
        'Value': [accuracy, precision, recall, specificity, f1, roc_auc]
    })

    # Generate ROC curve values using probabilities
    fpr, tpr, _ = roc_curve(y_test_target, y_prob_target)
    roc_auc_val = auc(fpr, tpr)

    # Plotting the ROC curve
    roc_auc_figure = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    return metrics_df, cm_figure, roc_auc_figure

def on_evaluate(model, dataset, test_dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    # Get test predictions
    y_test, logits, smiles_list = test(model, dataset, test_dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
    probabilities = torch.sigmoid(logits)
    y_test_np = y_test.int().detach().numpy()
    y_pred_np = torch.round(probabilities).int().detach().numpy()
    
    probabilities_np = probabilities.detach().numpy()
    eval_df = pd.DataFrame()
    eval_df['SMILES'] = smiles_list
    for col_idx, col_name in enumerate(dataset.target_column_names):
        eval_df[col_name] = y_test_np[:, col_idx].tolist()
        eval_df[f"{col_name} (Prediction)"] = y_pred_np[:, col_idx].tolist()
        eval_df[f"{col_name} (Probability)"] = probabilities_np[:, col_idx].tolist()

    status = f"Evaluation completed."

    return f"<span style='color: green;'>{status}</span>", eval_df, gr.update(interactive=True)

def on_export_evaluation(working_directory_path, file_name, eval_df):
    
    file_path = os.path.join(working_directory_path, f"{file_name}.csv")
    eval_df.to_csv(file_path)
    status = f"Evaluation exported: {file_name}.csv"

    return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)

def on_process_prediction_data(working_directory_path, prediction_dataset_file, dataset, prediction_graph_directory, prediction_molecular_fingerprint_directory, datatype, batch_size):
    try:
        # Process the prediction dataset and create a dataloader
        prediction_dataset_file_path = os.path.join(working_directory_path, prediction_dataset_file)
        prediction_graph_directory_path = os.path.join(working_directory_path, prediction_graph_directory)
        prediction_molecular_fingerprint_directory_path =  os.path.join(working_directory_path, prediction_molecular_fingerprint_directory)                                            
        prediction_dataset = MolecularDataset(prediction_dataset_file_path, dataset.load_data, prediction_graph_directory_path, prediction_molecular_fingerprint_directory_path, datatype)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size)
        
        status = "Prediction data processed successfully."
        return f"<span style='color: green;'>{status}</span>", prediction_dataset, prediction_dataloader, gr.update(interactive=True)
    except Exception as exc:
        status = f"Error processing prediction data: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, gr.update(interactive=False)

# Define the predict function
@torch.no_grad()
def predict(model, dataset, dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    smiles_list = []
    logits_list = []
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    
    for batch in dataloader:
        # Get input data
        if dataset.load_data == 'Graphs' or dataset.load_data == 'Both':
            x = batch.x.to(device=device, dtype=datatype)
            edge_index = batch.edge_index.long().to(device=device)
            edge_attr = batch.edge_attr.to(device=device, dtype=datatype)
            batch_index = batch.batch.to(device=device)
        else:
            x = None
            edge_index = None
            edge_attr = None
            batch_index = None
        if dataset.load_data == 'Molecular fingerprints' or dataset.load_data == 'Both':
            fingerprints = batch.fp
            if variance_threshold is not None and pca is not None:
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints)
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None
        
        with torch.no_grad():
            # Forward pass
            if dataset.load_data == 'Molecular fingerprints':
                output = model(fingerprints_tensor)
            elif need_edge_attr:
                output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
            else:
                output = model(x, edge_index, batch_index, fingerprints_tensor)
        
        smiles_list.extend(batch.smiles)
        logits_list.append(output.cpu().detach())

    logits = torch.cat(logits_list, dim=0) if logits_list else torch.tensor([])

    return smiles_list, logits

def on_predict(model, dataset, prediction_dataset, predict_dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    # Get predictions
    smiles_list, logits = predict(model, prediction_dataset, predict_dataloader, device, datatype, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
    probabilities = torch.sigmoid(logits)
    y_pred_np = torch.round(probabilities).int().detach().numpy()
    
    prediction_df = pd.DataFrame()
    prediction_df['SMILES'] = smiles_list
    for col_idx, col_name in enumerate(dataset.target_column_names):
        prediction_df[f"{col_name} (Prediction)"] = y_pred_np[:, col_idx].tolist()

    status = f"Prediction completed."

    return f"<span style='color: green;'>{status}</span>", prediction_df, gr.update(interactive=True)

def on_export_prediction(working_directory_path, prediction_file_name, prediction_df):
    prediction_file_path = os.path.join(working_directory_path, f"{prediction_file_name}.csv")
    prediction_df.to_csv(prediction_file_path)
    status = f"Predictions exported: {prediction_file_name}.csv"
    return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)

def binary_classification_tab_content():
    with gr.Tab("Binary Classification") as binary_classification_tab:
        with gr.Row():
            with gr.Column(scale=1):
                working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(), value="wd", allow_custom_value=True)
                working_directory_path_state = gr.State()
                open_working_directory_button = gr.Button(value="Create/Open Working Directory")
                working_directory_content_list_state = gr.State()
                working_directory_content_list_dataframe = gr.Dataframe(label="Contents in Working Directory", headers=["Name", "Type", "Modified"], max_height=600, interactive=False)
            with gr.Column(scale=3):
                with gr.Row(min_height=40):
                    status_html = gr.HTML()
                with gr.Accordion("Settings", open=False):
                    with gr.Row():
                        device_dropdown = get_device_dropdown()
                        datatype_dropdown = gr.Dropdown(label="Data type", value="float32", choices=["float16", "float32", "float64", "bfloat16"])
                        datatype_state = gr.State(value=torch.float32)
                        random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Random seed")
                with gr.Accordion("Upload Dataset"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Dataset file")
                            dataset_name_textbox = gr.Textbox(label="Dataset name", value="dataset")
                            smiles_column_textbox = gr.Textbox(label="SMILES column", value="SMILES")
                            target_columns_textbox = gr.Textbox(label="Target columns (separated by commas)", value="target")
                            upload_dataset_button = gr.Button(value="Load dataset", interactive=False)
                        with gr.Column(scale=2):
                            dataset_preview_dataframe = gr.Dataframe(label="Dataset preview", max_height=360, interactive=False)
                with gr.Accordion("Feature Extraction"):
                    with gr.Row():
                        dataset_file_feature_extraction_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Graph extraction**")
                            graph_directory_textbox = gr.Textbox(label="Graph directory", value="graphs")
                            graph_featurizer_dropdown = gr.Dropdown(label="Graph featurizer", value="MolGraphConvFeaturizer", choices=["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"])
                            extract_graphs_button = gr.Button(value="Extract graphs", interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Molecular fingerprint extraction**")
                            molecular_fingerprint_directory_textbox = gr.Textbox(label="Molecular fingerprint directory", value="molecular_fingerprints")
                            molecular_fingerprint_dropdown = gr.Dropdown(label="Molecular fingerprint", value="Morgan fingerprint", choices=["Morgan fingerprint", "Atom-pairs fingerprint", "Topological-torsion fingerprint", "Avalon fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint"])
                            radius_slider = gr.Slider(label="Radius", value=2, minimum=1, maximum=5, step=1)
                            number_of_bits_slider = gr.Slider(label="Number of bits", value=2048, minimum=128, maximum=4096, step=128)
                            extract_molecule_fingerprints_button = gr.Button(value="Extract molecular fingerprints", interactive=False)
                with gr.Accordion("Load and process data"):
                    with gr.Row():
                        dataset_file_process_data_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                        load_data_radio = gr.Radio(label="Which data to load?", choices=["Graphs", "Molecular fingerprints", "Both"], value="Both")
                    with gr.Row():
                        with gr.Column(scale=1):
                            graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[])
                            molecular_fingerprint_directory_dropdown = gr.Dropdown(label="Molecular fingerprint directory", choices=[])
                            dimensionality_reduction_checkbox = gr.Checkbox(label="Dimensionality reduction (for molecular fingerprints)", value=False)
                            variance_threshold_slider = gr.Slider(minimum=0, maximum=1.0, value=0.01, step=0.01, label="Variance threshold", visible=False)
                            pca_num_components_slider = gr.Slider(minimum=1, maximum=256, value=32, step=1, label="PCA number of components", visible=False)
                        with gr.Column(scale=1):
                            test_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Test size")
                            val_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Validation size")
                            batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
                            process_data_button = gr.Button(value="Process data", interactive=False)
                            dataset_state = gr.State()
                            train_dataloader_state = gr.State()
                            val_dataloader_state = gr.State()
                            test_dataloader_state = gr.State()
                            variance_threshold_state = gr.State()
                            pca_state = gr.State()
                with gr.Accordion("Model"):
                    with gr.Row():
                        with gr.Column(scale=1) as graph_convolutional_block:
                            gr.Markdown(value="**Graph convolutional block**")
                            gnn_model_tab_state = gr.State(value="GCN")
                            with gr.Tabs():
                                with gr.Tab("GCN") as gcn_tab:
                                    gcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    gcn_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    gcn_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    gcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    gcn_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("GraphSAGE") as graph_sage_tab:
                                    graph_sage_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    graph_sage_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    graph_sage_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    graph_sage_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    graph_sage_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("GIN") as gin_tab:
                                    gin_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    gin_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    gin_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    gin_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    gin_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("GAT") as gat_tab:
                                    gat_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    gat_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    gat_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    gat_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    gat_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("EdgeCNN") as edgecnn_tab:
                                    edgecnn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    edgecnn_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    edgecnn_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    edgecnn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    edgecnn_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("AttentiveFP") as attentivefp_tab:
                                    attentivefp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    attentivefp_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                    attentivefp_num_timesteps = gr.Slider(label="num_timesteps", value=3, minimum=1, maximum=10, step=1)
                                    attentivefp_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                    attentivefp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    attentivefp_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("Custom") as customgcn_tab:
                                    customgcn_convolutional_layer_name = gr.Dropdown(label="Graph convolutional layer", value="GCNConv", choices=["GCNConv", "SAGEConv", "CuGraphSAGEConv", "SGConv", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "SuperGATConv", "TransformerConv", "GeneralConv"])
                                    customgcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    customgcn_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                                    customgcn_n_heads = gr.Slider(label="n_heads", minimum=1, maximum=8, value=3, step=1)
                                    customgcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    customgcn_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                        with gr.Column(scale=1) as molecular_fingerprint_block:
                            gr.Markdown(value="**Molecular fingerprint block**")
                            mlp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                            mlp_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                            mlp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Predictor block**")
                            predictor_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                            predictor_n_layers = gr.Slider(label="n_layers", minimum=0, maximum=6, value=1, step=1)
                    with gr.Row():
                        create_model_button = gr.Button(value="Create model", interactive=False)
                        model_state = gr.State()
                        train_losses_state = gr.State()
                        val_losses_state = gr.State()
                        trained_epochs_state = gr.State()
                    with gr.Row():
                        with gr.Column(scale=2):
                            model_summary_textarea = gr.TextArea(label="Model summary", elem_classes="monospaced")
                        with gr.Column(scale=1):
                            checkpoint_name_textbox = gr.Textbox(label="Checkpoint name", value="checkpoint")
                            save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
                            checkpoint_file_dropdown = gr.Dropdown(label="Checkpoint file", choices=[])
                            load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
                with gr.Accordion("Model training"):
                    with gr.Row(): 
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown(value="Optimizer")
                                optimizer_dropdown = gr.Dropdown(label="Optimizer", value="Adam", choices=["Adam", "AdamW", "SGD"]) 
                                learning_rate_slider = gr.Slider(label="Learning rate", minimum=0.000001, maximum=0.1, value=0.0001, step=0.000001)
                                learning_rate_decay_slider = gr.Slider(label="Learning rate decay", minimum=0.5, maximum=1, value=0.99, step=0.001)
                            create_optimizer_button = gr.Button(value="Create optimizer", interactive=False)
                            optimizer_state = gr.State()
                            schedular_state = gr.State()
                        with gr.Column(scale=3):
                            gr.Markdown(value="Training")  
                            with gr.Row(equal_height=True): 
                                with gr.Column(scale=1):
                                    epochs_slider = gr.Slider(label="Epochs", minimum=1, maximum=1000, value=100, step=1)
                                with gr.Column(scale=1):
                                    train_button = gr.Button(value="Train", interactive=False)
                            with gr.Row(): 
                                with gr.Column(scale=3):
                                    training_plot = gr.Plot()
                                with gr.Column(scale=1):
                                    loss_file_name_textbox = gr.Textbox(label="File name", value="losses")
                                    export_losses_button = gr.Button(value="Export", interactive=False)
                with gr.Accordion("Model evaluation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            evaluate_button = gr.Button(value="Evaluate", interactive=False)
                            target_column_dropdown = gr.Dropdown(label="Output", choices=[]) 
                            evaluation_metrics_dataframe = gr.DataFrame(label="Evaluation metrics", wrap=False, interactive=False)
                            evaluation_file_name_textbox = gr.Textbox(label="File name", value="evaluation")
                            export_evaluation_button = gr.Button(value="Export", interactive=False)
                            evaluation_df_state = gr.State()
                        with gr.Column(scale=1):
                            confusion_matrix_plot = gr.Plot()
                        with gr.Column(scale=1):
                            roc_auc_curve_plot = gr.Plot()
                with gr.Accordion("Make prediction"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_dataset_file_dropdown = gr.Dropdown(label="Prediction dataset file", choices=[])
                            prediction_graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[])
                            prediction_molecular_fingerprint_directory_dropdown = gr.Dropdown(label="Molecular fingerprint directory", choices=[])
                            process_prediction_data_button = gr.Button(value="Process prediction data", interactive=False)
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=True): 
                                with gr.Column(scale=1):
                                    predict_button = gr.Button(value="Predict", interactive=False)
                                    prediction_dataset_state = gr.State()
                                    prediction_dataloader_state = gr.State()
                                with gr.Column(scale=1):
                                    prediction_file_name_textbox = gr.Textbox(label="File name", value="prediction")
                                    export_prediction_button = gr.Button(value="Export", interactive=False)
                            with gr.Row(): 
                                prediction_dataframe = gr.DataFrame(label="Prediction", wrap=False, interactive=False)

    # Working directory interactions
    working_directory_dropdown.change(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, upload_dataset_button, extract_graphs_button, extract_molecule_fingerprints_button, process_data_button, working_directory_path_state, working_directory_content_list_state])
    open_working_directory_button.click(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, upload_dataset_button, extract_graphs_button, extract_molecule_fingerprints_button, process_data_button, working_directory_path_state, working_directory_content_list_state])
    working_directory_content_list_state.change(on_content_list_change, [working_directory_path_state, dataset_name_textbox, graph_directory_textbox, molecular_fingerprint_directory_textbox, checkpoint_name_textbox],
                                                [working_directory_content_list_dataframe,
                                                 dataset_file_feature_extraction_dropdown, graph_directory_dropdown, molecular_fingerprint_directory_dropdown, dataset_file_process_data_dropdown, checkpoint_file_dropdown, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown])
    
    # Settings interaction
    datatype_dropdown.change(on_datatype_change, datatype_dropdown, datatype_state)

    # Load dataset and process data interactions
    upload_dataset_button.click(on_upload_dataset, [dataset_file, working_directory_path_state, dataset_name_textbox, smiles_column_textbox, target_columns_textbox], [status_html, dataset_preview_dataframe, working_directory_content_list_state])

    # Feature extraction interactions
    extract_graphs_button.click(on_extract_graphs, [working_directory_path_state, dataset_file_feature_extraction_dropdown, graph_directory_textbox, graph_featurizer_dropdown, datatype_state], [status_html, working_directory_content_list_state])
    molecular_fingerprint_dropdown.change(on_molecular_fingerprint_change, molecular_fingerprint_dropdown, radius_slider)
    load_data_radio.change(on_load_data_option_change, load_data_radio, [graph_directory_dropdown, molecular_fingerprint_directory_dropdown, dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider, graph_convolutional_block, molecular_fingerprint_block, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown])
    extract_molecule_fingerprints_button.click(on_extract_molecule_fingerprints, [working_directory_path_state, dataset_file_feature_extraction_dropdown, molecular_fingerprint_directory_textbox, molecular_fingerprint_dropdown, radius_slider, number_of_bits_slider], [status_html, working_directory_content_list_state])

    # Process data interactions
    dimensionality_reduction_checkbox.change(on_dimensionality_reduction_change, dimensionality_reduction_checkbox, [variance_threshold_slider, pca_num_components_slider])
    process_data_button.click(on_process_data, [working_directory_path_state, dataset_file_process_data_dropdown, load_data_radio, graph_directory_dropdown, molecular_fingerprint_directory_dropdown, datatype_state, dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider, test_size_slider, val_size_slider, batch_size_dropdown, random_seed_slider],
                              [status_html, create_model_button, target_column_dropdown, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state, variance_threshold_state, pca_state])
    
    # Create model interactions
    create_model_button.click(on_create_model, [dataset_state, gnn_model_tab_state, device_dropdown, datatype_state, random_seed_slider, variance_threshold_state, pca_state,
                                                gcn_n_hiddens, gcn_num_layers, gcn_dropout, gcn_n_outputs,
                                                graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_dropout, graph_sage_n_outputs,
                                                gin_n_hiddens, gin_num_layers, gin_dropout, gin_n_outputs,
                                                gat_n_hiddens, gat_num_layers, gat_dropout, gat_n_outputs,
                                                edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_dropout, edgecnn_n_outputs,
                                                attentivefp_n_hiddens, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_dropout, attentivefp_n_outputs,
                                                customgcn_convolutional_layer_name, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                                mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                predictor_n_hiddens, predictor_n_layers],
                                                [status_html, model_summary_textarea, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, process_prediction_data_button, model_state, train_losses_state, val_losses_state, trained_epochs_state])
    save_checkpoint_button.click(on_save_checkpoint, [working_directory_path_state, checkpoint_name_textbox, model_state, trained_epochs_state], [status_html, working_directory_content_list_state])
    load_checkpoint_button.click(on_load_checkpoint, [working_directory_path_state, checkpoint_file_dropdown, model_state], [status_html, model_state, train_losses_state, val_losses_state, trained_epochs_state])
    
    # Train model interactions
    create_optimizer_button.click(on_create_optimizer, [model_state, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [status_html, optimizer_state, schedular_state, train_button])
    train_button.click(on_train, [model_state, dataset_state, train_dataloader_state, val_dataloader_state, device_dropdown, datatype_state,
                                  optimizer_state, schedular_state, epochs_slider, trained_epochs_state, train_losses_state, val_losses_state,
                                  gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                  [status_html, training_plot, export_losses_button, trained_epochs_state, train_losses_state, val_losses_state])
    export_losses_button.click(on_export_losses, [working_directory_path_state, loss_file_name_textbox, train_losses_state, val_losses_state], [status_html, working_directory_content_list_state])
    
    # Evaluate model interactions
    evaluate_button.click(on_evaluate, [model_state, dataset_state, test_dataloader_state, device_dropdown, datatype_state,
                                        gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                        [status_html, evaluation_df_state, export_evaluation_button]
                         ).then(on_target_column_change, [evaluation_df_state, target_column_dropdown], [evaluation_metrics_dataframe, confusion_matrix_plot, roc_auc_curve_plot])
    target_column_dropdown.change(on_target_column_change, [evaluation_df_state, target_column_dropdown], [evaluation_metrics_dataframe, confusion_matrix_plot, roc_auc_curve_plot])
    export_evaluation_button.click(on_export_evaluation, [working_directory_path_state, evaluation_file_name_textbox, evaluation_df_state], [status_html, working_directory_content_list_state])
        
    # Make prediction interactions
    process_prediction_data_button.click(on_process_prediction_data, [working_directory_path_state, prediction_dataset_file_dropdown, dataset_state, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown, datatype_state, batch_size_dropdown], [status_html, prediction_dataset_state, prediction_dataloader_state, predict_button])
    predict_button.click(on_predict, [model_state, dataset_state, prediction_dataset_state, prediction_dataloader_state, device_dropdown, datatype_state,
                                      gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                      [status_html, prediction_dataframe, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [working_directory_path_state, prediction_file_name_textbox, prediction_dataframe], [status_html, working_directory_content_list_state])
    
    return binary_classification_tab