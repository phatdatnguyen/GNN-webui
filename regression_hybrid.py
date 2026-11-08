import math
import os
from html import escape
import time
import pandas as pd
import numpy as np
import gradio as gr
import torch
from torch_geometric.nn import summary
from torch_geometric.loader import DataLoader
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from evaluation_metrics import metrics_from_predictions
from matplotlib import pyplot as plt
from datasets import MolecularDataset
from ui_utils import bind_data_processing, bind_result_invalidation, dataset_cleanup_controls
from gnn_run_ui import attach_model_run, capture_split_outputs, route_model_export, bind_working_directory, load_run_checkpoint, ensure_model_split, directory_reset_updates, show_model_run, capture_feature_sources, bind_artifact_update
from run_utils import make_split_context, ensure_run_directory, list_saved_models, resolve_saved_model, validate_seed
from similarity_ui import build_similarity_controls, bind_similarity_controls
from models import *
from utils import *

def get_working_directories():
    workflow_directory = os.path.join(DATA_DIR, "Regression")
    os.makedirs(workflow_directory, exist_ok=True)
    return sorted(d for d in os.listdir(workflow_directory) if os.path.isdir(os.path.join(workflow_directory, d)))

def get_working_directory_contents(working_directory_path):
    # os.listdir(None) silently lists the current working directory, which would show the
    # repo root in the file browser after a failed "open working directory".
    if not working_directory_path:
        return []
    try:
        return os.listdir(working_directory_path)
    except OSError:
        return []

def on_open_working_directory(working_directory_name):
    try:
        working_directory_name = sanitize_path_component(working_directory_name, "working directory name")
    except ValueError as exc:
        gr.Warning(str(exc))
        return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), None, None

    working_directory_path = os.path.join(DATA_DIR, "Regression", working_directory_name)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_working_directory_contents(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory_name), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), working_directory_path, files

def on_content_list_change(working_directory_path, dataset_file_name, graph_directory, molecular_fingerprint_directory, checkpoint_file_name):
    if not working_directory_path:
        # Reachable when opening a working directory failed and the path state was cleared.
        empty = gr.update(choices=[], value=None)
        return pd.DataFrame(columns=["File", "Type", "Modified"]), empty, empty, empty, empty, empty, empty, empty, empty

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
        modified_timestamp = os.path.getmtime(content_path)
        contents_info.append([content, file_type, time.ctime(modified_timestamp), modified_timestamp])
    # Sort on the numeric timestamp: the ctime string sorts by weekday name, not by date.
    contents_info.sort(key=lambda x: x[3], reverse=True)
    contents_df = pd.DataFrame([row[:3] for row in contents_info], columns=["File", "Type", "Modified"])

    # Filter files
    data_files = [f for f in contents if f.endswith('.csv')]
    graph_directories = list_feature_directories(working_directory_path, "graph")
    fingerprint_directories = list_feature_directories(working_directory_path, "fingerprint")
    try:
        graph_directory = prefixed_feature_directory(graph_directory, "graph")
    except ValueError:
        graph_directory = None
    try:
        molecular_fingerprint_directory = prefixed_feature_directory(molecular_fingerprint_directory, "fingerprint")
    except ValueError:
        molecular_fingerprint_directory = None
    checkpoint_files = list_saved_models(working_directory_path, ".ckpt")

    # Update the file dropdown choices and value
    if f"{dataset_file_name}.csv" in data_files:
        dataset_file_name_value = f"{dataset_file_name}.csv"
    else:
        dataset_file_name_value = data_files[0] if data_files else None

    if graph_directory in graph_directories:
        graph_directory_value = graph_directory
    else:
        graph_directory_value = graph_directories[0] if graph_directories else None

    if molecular_fingerprint_directory in fingerprint_directories:
        molecular_fingerprint_directory_value = molecular_fingerprint_directory
    else:
        molecular_fingerprint_directory_value = fingerprint_directories[0] if fingerprint_directories else None

    if checkpoint_file_name in checkpoint_files:
        checkpoint_file_name_value = checkpoint_file_name
    else:
        checkpoint_file_name_value = checkpoint_files[0] if checkpoint_files else None

    if f"{dataset_file_name}.csv" in data_files:
        prediction_dataset_file_name_value = f"{dataset_file_name}.csv"
    else:
        prediction_dataset_file_name_value = data_files[0] if data_files else None

    if graph_directory in graph_directories:
        prediction_graph_directory_value = graph_directory
    else:
        prediction_graph_directory_value = graph_directories[0] if graph_directories else None

    if molecular_fingerprint_directory in fingerprint_directories:
        prediction_molecular_fingerprint_directory_value = molecular_fingerprint_directory
    else:
        prediction_molecular_fingerprint_directory_value = fingerprint_directories[0] if fingerprint_directories else None

    return contents_df, \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=graph_directories, value=graph_directory_value), \
           gr.update(choices=fingerprint_directories, value=molecular_fingerprint_directory_value), \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=checkpoint_files, value=checkpoint_file_name_value), \
           gr.update(choices=data_files, value=prediction_dataset_file_name_value), \
           gr.update(choices=graph_directories, value=prediction_graph_directory_value), \
           gr.update(choices=fingerprint_directories, value=prediction_molecular_fingerprint_directory_value)

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

def on_upload_dataset(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns,
                      canonicalize_smiles=True, remove_duplicates=True, remove_salt=True):
    try:
        if not working_directory_path or not os.path.isdir(working_directory_path):
            raise ValueError("Please open a working directory first.")
        if dataset_file_path is None or not str(dataset_file_path).strip():
            raise ValueError("Please select a dataset file.")
        status, df = process_dataset_file(
            dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns,
            canonicalize_smiles=canonicalize_smiles, remove_duplicates=remove_duplicates, remove_salt=remove_salt)
        if df is None:
            return status, gr.update(), gr.update()
        return status, df.head(10), get_working_directory_contents(working_directory_path)
    except Exception as exc:
        return f"<span style='color: red;'>Error uploading dataset: {escape(str(exc))}</span>", gr.update(), gr.update()

def on_extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress=gr.Progress()):
    status = extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress)
    return status, get_working_directory_contents(working_directory_path)

def on_molecular_fingerprint_change(molecular_fingerpint_name):
    return gr.update(visible=(molecular_fingerpint_name == 'Morgan fingerprint'))

def on_load_data_option_change(load_data_option, dimensionality_reduction_enabled):
    load_graphs = (load_data_option == 'Graphs' or load_data_option == 'Both')
    load_fingerprints = (load_data_option == 'Molecular fingerprints' or load_data_option == 'Both')
    # The two PCA sliders belong to the checkbox, so they stay hidden unless it is ticked.
    show_reduction = load_fingerprints and dimensionality_reduction_enabled
    return gr.update(visible=load_graphs), gr.update(visible=load_fingerprints), gr.update(visible=load_fingerprints), gr.update(visible=show_reduction), gr.update(visible=show_reduction), gr.update(visible=load_graphs), gr.update(visible=load_fingerprints), gr.update(visible=load_graphs), gr.update(visible=load_fingerprints)

def on_extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress=gr.Progress()):
    status = extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress)
    return status, get_working_directory_contents(working_directory_path)

def on_dimensionality_reduction_change(dimensionality_reduction_enabled):
    return gr.update(visible=dimensionality_reduction_enabled), gr.update(visible=dimensionality_reduction_enabled)

def on_process_data(working_directory_path, dataset_file_name, load_data, graph_directory, molecular_fingerprint_directory, datatype, dimensionality_reduction, variance_threshold, pca_num_components, test_ratio, val_ratio, batch_size, shuffle_training_data, random_seed, split_method="Random"):
    try:
        # Validate inputs
        random_seed = validate_seed(random_seed)
        if load_data in ('Graphs', 'Both') and (graph_directory is None or graph_directory.strip() == ""):
            raise ValueError("Please select a graph directory.")
        if load_data in ('Molecular fingerprints', 'Both') and (molecular_fingerprint_directory is None or molecular_fingerprint_directory.strip() == ""):
            raise ValueError("Please select a molecular fingerprint directory.")

        dataset_file_name = sanitize_path_component(dataset_file_name, "dataset file name")

        # Validate output columns
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        target_columns_list = df.drop(columns=['SMILES']).columns.tolist()
        if len(target_columns_list) == 0:
            raise ValueError("No target columns found in the dataset. Please ensure your dataset has at least one target column in addition to the SMILES column.")
        for col in target_columns_list:
            if not pd.api.types.is_numeric_dtype(df[col]) :
                raise ValueError(f"Target column '{col}' is not numeric. Please ensure all target columns are numeric.")
            if not np.isfinite(df[col].to_numpy()).all():
                raise ValueError(f"Target column '{col}' contains missing or non-finite values. Please provide finite values for every molecule.")

        # Set up dataset and dataloaders
        graph_directory_path = (os.path.join(working_directory_path, sanitize_path_component(graph_directory, "graph directory"))
                                if load_data in ('Graphs', 'Both') else None)
        molecular_fingerprint_directory_path = (os.path.join(working_directory_path, sanitize_path_component(molecular_fingerprint_directory, "molecular fingerprint directory"))
                                                if load_data in ('Molecular fingerprints', 'Both') else None)

        # Preserve raw target precision until normalization, including low-precision runs.
        dataset = MolecularDataset(dataset_file_path, load_data, graph_directory_path, molecular_fingerprint_directory_path, datatype, target_datatype=torch.float64)
        
        total_len = len(dataset)
        capture_feature_sources(dataset, graph_directory_path, molecular_fingerprint_directory_path)
        train_dataset, val_dataset, test_dataset = split_molecular_dataset(
            dataset, test_ratio, val_ratio, random_seed, split_method)
        dataset.split_context = make_split_context(
            working_directory_path, dataset_file_name, df, train_dataset.indices, test_dataset.indices,
            random_seed, split_method, validation_indices=val_dataset.indices)

        
        # Re-shuffle the training set each epoch. The generator is seeded from the random seed
        # so a shuffled run is still reproducible; turn the checkbox off for a fixed batch order.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=bool(shuffle_training_data),
                                      generator=torch.Generator().manual_seed(random_seed))
        # Validation and test order does not affect the metrics, so leave them deterministic.
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Set up feature transformers (fit on training split only to avoid leakage).
        # The checkbox is only hidden - not reset - when fingerprints are not loaded, so ignore it here.
        if dimensionality_reduction and load_data in ('Molecular fingerprints', 'Both'):
            variance_threshold = VarianceThreshold(variance_threshold)
            pca = PCA(n_components=pca_num_components, random_state=random_seed)
            fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, "molecular_fingerprints.csv"))
            train_fingerprints = fingerprint_df.drop(columns=['SMILES']).iloc[train_dataset.indices].values
            train_fingerprints_variance_thresholded = variance_threshold.fit_transform(train_fingerprints)
            pca.fit(train_fingerprints_variance_thresholded)
        else:
            variance_threshold = None
            pca = None

        # Set up output scaler (fit on training split only to avoid leakage)
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        train_targets = dataset.df.drop(columns=['SMILES']).iloc[train_dataset.indices].to_numpy()
        output_scaler.fit(train_targets)

        target_column_names = dataset.target_column_names

        status = (f"Data processed: {total_len} molecules. {split_method} split: "
                  f"{len(train_dataset)} training, {len(val_dataset)} validation, "
                  f"{len(test_dataset)} test.")
        if split_method == "Scaffold":
            status += " Scaffold groups are kept together, so split sizes may differ from the requested ratios."
        status += " Create a model for this split before training or evaluation."
        return f"<span style='color: green;'>{status}</span>", gr.update(interactive=True), gr.update(choices=target_column_names, value=target_column_names[0]), dataset, train_dataloader, val_dataloader, test_dataloader, variance_threshold, pca, output_scaler
    except Exception as exc:
        status = f"Error processing data: {exc}"
        return f"<span style='color: red;'>{status}</span>", gr.update(interactive=False), gr.update(choices=[], value=None), None, None, None, None, None, None, None

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
                    predictor_n_hiddens, predictor_n_layers, model_name=None):
    
    run_parameters = locals().copy()
    model = None
    train_losses = []
    val_losses = []
    trained_epochs = 0

    try:
        if dataset is None:
            raise ValueError("Please process the data first.")
        random_seed = validate_seed(random_seed)
        run_parameters["random_seed"] = random_seed
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
            model = GATModel(gcn_n_inputs, gat_n_hiddens, edge_dim, gat_num_layers, gat_n_outputs, gat_dropout,
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
            elif customgcn_convolutional_layer_name == "ChebConv":
                model = ChebConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
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
            elif customgcn_convolutional_layer_name == "FeaStConv":
                model = FeaStConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
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
                model = GATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
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
        
        batch_index = torch.zeros(x.size(0), device=device, dtype=torch.long) if x is not None else None
        
        if fingerprint is not None:
            if variance_threshold is not None and pca is not None:
                fingerprint_variance_thresholded = variance_threshold.transform(fingerprint.reshape(1, -1).double().numpy())
                fingerprint_pca = pca.transform(fingerprint_variance_thresholded)
                fingerprint_tensor = torch.tensor(fingerprint_pca).to(device=device, dtype=datatype)
            else:
                fingerprint_tensor = torch.tensor(fingerprint.reshape(1, -1)).to(device=device, dtype=datatype)
        else:
            fingerprint_tensor = None
        
        # Model summary
        need_edge_attr = model_requires_edge_attr(model)
        if dataset.load_data == 'Molecular fingerprints':
            model_summary = summary(model, fingerprint_tensor)
        elif need_edge_attr:
            model_summary = summary(model, x, edge_index, edge_attr, batch_index, fingerprint_tensor)
        else:
            model_summary = summary(model, x, edge_index, batch_index, fingerprint_tensor)
        
        attach_model_run(model, dataset, model_name, random_seed, "regression_hybrid", run_parameters)
        status = "Model created successfully. Create an optimizer to start training."
        # Training is disabled until an optimizer is created for *this* model: an optimizer
        # captures the parameters it was built with, so a stale one silently updates nothing.
        return f"<span style='color: green;'>{status}</span>", model_summary, \
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), \
            model, train_losses, val_losses, trained_epochs
    except Exception as exc:
        status = f"Error creating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, \
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), \
            model, train_losses, val_losses, trained_epochs

def on_save_checkpoint(working_directory_path, checkpoint_name, model, trained_epochs, variance_threshold, pca, output_scaler):
    try:
        if model is None:
            raise ValueError("Please create a model first.")
        checkpoint_name = sanitize_path_component(checkpoint_name, "checkpoint name")
        # The fitted transformers are part of the model's contract. In particular the output scaler
        # denormalises every prediction, so a checkpoint reloaded without it silently returns
        # predictions on the wrong scale.
        checkpoint = {
            'model_class': type(model).__name__,
            'state_dict': model.state_dict(),
            'run_context': getattr(model, '_run_context', None),
            'trained_epochs': trained_epochs,
            'variance_threshold': variance_threshold,
            'pca': pca,
            'output_scaler': output_scaler
        }
        run_context = getattr(model, '_run_context', None)
        if run_context is not None and (not working_directory_path or os.path.realpath(working_directory_path) != os.path.realpath(run_context['working_directory'])):
            raise ValueError("Please reopen this model's working directory before saving its checkpoint.")
        output_directory = ensure_run_directory(run_context) if run_context is not None else working_directory_path
        checkpoint_path = os.path.join(output_directory, f"{checkpoint_name}_{trained_epochs}.ckpt")
        if os.path.islink(checkpoint_path):
            raise ValueError("Checkpoint files cannot be symbolic links.")
        # The filename is derived from the epoch count, so re-saving replaces a previous file.
        replaced = os.path.exists(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        status = f"Model checkpoint saved: {checkpoint_name}_{trained_epochs}.ckpt"
        if replaced:
            status += " (replaced an existing file of the same name)"
        return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error saving checkpoint: {exc}"
        return f"<span style='color: red;'>{status}</span>", get_working_directory_contents(working_directory_path)

def on_load_checkpoint(working_directory_path, checkpoint_file_name, model, train_losses, val_losses, trained_epochs, variance_threshold, pca, output_scaler):
    try:
        if model is None:
            raise ValueError("Please create a model first, so the checkpoint has an architecture to load into.")
        checkpoint_file_path = resolve_saved_model(working_directory_path, checkpoint_file_name, ".ckpt")
        checkpoint, run_note = load_run_checkpoint(model, checkpoint_file_path, checkpoint_file_name, working_directory_path)

        status = f"Checkpoint loaded: {checkpoint_file_name}.{run_note}"
        return f"<span style='color: green;'>{status}</span>", model, [], [], checkpoint.get('trained_epochs', 0), \
            checkpoint.get('variance_threshold', variance_threshold), checkpoint.get('pca', pca), \
            checkpoint.get('output_scaler', output_scaler)
    except Exception as exc:
        status = f"Error loading checkpoint: {exc}"
        # Leave the existing training state untouched - returning None here breaks the next Train click.
        return f"<span style='color: red;'>{status}</span>", model, train_losses, val_losses, trained_epochs, variance_threshold, pca, output_scaler

loss_fn = torch.nn.MSELoss()
def on_create_optimizer(model, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider):
    try:
        if model is None:
            raise ValueError("Please create a model first.")
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD
        }
        opt_class = optimizers.get(optimizer_dropdown)
        if opt_class is None:
            raise ValueError(f"Unknown optimizer '{optimizer_dropdown}'.")

        optimizer = opt_class(model.parameters(), lr=learning_rate_slider)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay_slider)

        status = "Optimizer created."
        return f"<span style='color: green;'>{status}</span>", optimizer, scheduler, gr.update(interactive=True)
    except Exception as exc:
        status = f"Error creating optimizer: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, gr.update(interactive=False)

# Define the train function
def train(model, dataset, dataloader, device, datatype, output_scaler, optimizer, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.train()
    total_loss = 0.0
    num_samples = 0
    need_edge_attr = model_requires_edge_attr(model)
    
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
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints.double().numpy())
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y.double().numpy()), device=device, dtype=datatype)
        
        # Forward pass
        optimizer.zero_grad()
        if dataset.load_data == 'Molecular fingerprints':
            output = model(fingerprints_tensor)
        elif need_edge_attr:
            output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
        else:
            output = model(x, edge_index, batch_index, fingerprints_tensor)

        if output.shape != y_scaled.shape:
            raise ValueError("Model output shape does not match the targets. Please re-create the model for this dataset.")

        # Compute loss and backpropagate
        # CPU MSE kernels do not support bfloat16 backward; accumulate low-precision losses in float32.
        loss = loss_fn(output.float(), y_scaled.float()) if datatype in (torch.float16, torch.bfloat16) else loss_fn(output, y_scaled)
        if not torch.isfinite(loss):
            raise ValueError("Training loss is not finite. Try a lower learning rate or float32 data type.")
        loss.backward()
        optimizer.step()
        # Accumulate squared error and take the root once, so this is the true RMSE over the
        # epoch rather than a mean of per-batch RMSEs (which depends on the batch size).
        total_loss += loss.item() * y_scaled.shape[0]
        num_samples += y_scaled.shape[0]

    return math.sqrt(total_loss / num_samples) if num_samples > 0 else 0.0

# Define the validation function
@torch.no_grad()
def validation(model, dataset, dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    need_edge_attr = model_requires_edge_attr(model)
    
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
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints.double().numpy())
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y.double().numpy()), device=device, dtype=datatype)
        
        with torch.no_grad():
            # Forward pass
            if dataset.load_data == 'Molecular fingerprints':
                output = model(fingerprints_tensor)
            elif need_edge_attr:
                output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
            else:
                output = model(x, edge_index, batch_index, fingerprints_tensor)

            if output.shape != y_scaled.shape:
                raise ValueError("Model output shape does not match the targets. Please re-create the model for this dataset.")

            # Compute loss
            loss = loss_fn(output.float(), y_scaled.float()) if datatype in (torch.float16, torch.bfloat16) else loss_fn(output, y_scaled)
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite. Try a lower learning rate or float32 data type.")
            total_loss += loss.item() * y_scaled.shape[0]
        num_samples += y_scaled.shape[0]

    return math.sqrt(total_loss / num_samples) if num_samples > 0 else 0.0

def on_train(model, dataset, train_dataloader, val_dataloader, device, datatype, output_scaler, optimizer, scheduler, epochs, trained_epochs, train_losses, val_losses, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca, progress=gr.Progress()):
    try:
        ensure_optimizer_matches_model(model, optimizer)
        ensure_model_on_device(model, device)
        ensure_model_split(model, dataset)
        if train_dataloader is None or val_dataloader is None:
            raise ValueError("Please process the data first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data first.")

        # Training loop. Both losses are appended together and the epoch counter is bumped
        # only after a full epoch, so a failure part-way through leaves consistent state.
        for _ in progress.tqdm(range(trained_epochs, epochs + trained_epochs), total=epochs, desc="Training"):
            train_loss = train(model, dataset, train_dataloader, device, datatype, output_scaler, optimizer, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
            val_loss = validation(model, dataset, val_dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            trained_epochs += 1
            scheduler.step()

        status = f"<span style='color: green;'>Training completed for {trained_epochs} epochs.</span>"
    except Exception as exc:
        status = f"<span style='color: red;'>Error during training: {exc}</span>"

    # Plot whatever was completed, so a mid-run failure does not discard finished epochs.
    figure = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE loss (scaled targets)')
    plt.legend()
    plt.grid(True)
    plt.close(figure)

    export_losses_button = gr.Button(value="Export", interactive=bool(train_losses))

    return status, figure, export_losses_button, trained_epochs, train_losses, val_losses

def on_export_losses(working_directory_path, file_name, train_losses, val_losses):
    try:
        if not train_losses:
            raise ValueError("Please train the model first.")
        df = pd.DataFrame({"epoch": range(len(train_losses)), "train_losses": train_losses, "val_losses": val_losses})
        filename = export_result_csv(working_directory_path, file_name, df)
        status = f"Losses exported to {filename}."
        return f"<span style='color: green;'>{escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting losses: {exc}"
        return f"<span style='color: red;'>{escape(status)}</span>", gr.update()

# Define the test function
@torch.no_grad()
def test(model, dataset, dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    y_test_list = []
    y_pred_scaled_list = []
    smiles_arr = []
    need_edge_attr = model_requires_edge_attr(model)
    
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
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints.double().numpy())
                fingerprints_pca = pca.transform(fingerprints_variance_thresholded)
                fingerprints_tensor = torch.tensor(fingerprints_pca).to(device=device, dtype=datatype)
            else:
                fingerprints_tensor = fingerprints.to(device=device, dtype=datatype)
        else:
            fingerprints_tensor = None
        
        # Get target values
        y = batch.y
        smiles = batch.smiles

        with torch.no_grad():
            # Forward pass
            if dataset.load_data == 'Molecular fingerprints':
                output = model(fingerprints_tensor)
            elif need_edge_attr:
                output = model(x, edge_index, edge_attr, batch_index, fingerprints_tensor)
            else:
                output = model(x, edge_index, batch_index, fingerprints_tensor)

        smiles_arr.extend(smiles)
        y_test_list.append(y.detach())
        y_pred_scaled_list.append(output.cpu().detach())

    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.tensor([])
    y_pred_scaled = torch.cat(y_pred_scaled_list, dim=0) if y_pred_scaled_list else torch.tensor([])
    y_pred = output_scaler.inverse_transform(y_pred_scaled.double().numpy()) if y_pred_scaled.numel() else np.empty((0, dataset.num_outputs))
    return y_test, y_pred, smiles_arr

def on_target_column_change(eval_df, target_column):
    if eval_df is None or eval_df.empty or target_column is None:
        return None, None

    try:
        # Evaluate the model using different metrics
        y_test_target = eval_df[target_column].to_numpy()
        y_pred_target = eval_df[f"{target_column} (Prediction)"].to_numpy()
        metrics_df = metrics_from_predictions("regression", eval_df, [target_column]).drop(columns="Target")
        metrics_df["Value"] = metrics_df["Value"].round(4)

        # Visualization
        scatter_plot = plt.figure(figsize=(8, 8))
        plt.scatter(y_test_target, y_pred_target)
        limits = [min(y_test_target.min(), y_pred_target.min()), max(y_test_target.max(), y_pred_target.max())]
        plt.plot(limits, limits, color='navy', lw=1, linestyle='--')
        plt.xlabel('actual value')
        plt.ylabel('predicted value')
        plt.grid(True)
        plt.close(scatter_plot)

        return metrics_df, scatter_plot
    except Exception as exc:
        gr.Warning(f"Error computing metrics for '{target_column}': {exc}")
        return None, None

def on_evaluate(model, dataset, test_dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    try:
        if model is None or test_dataloader is None:
            raise ValueError("Please process the data and create a model first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data, or load a checkpoint that stores one.")
        columns = ["SMILES"]
        for target in dataset.target_column_names:
            columns.extend([target, f"{target} (Prediction)"])
        if len(columns) != len(set(columns)):
            raise ValueError("Target names overlap generated evaluation columns. Rename the affected targets and retrain.")
        ensure_model_on_device(model, device)
        ensure_model_split(model, dataset)
        # Get test predictions
        y_test, y_pred, test_smiles_arr = test(model, dataset, test_dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
        if len(test_smiles_arr) == 0:
            raise ValueError("The test split is empty. Please lower the validation size or raise the test size.")
        y_test_np = y_test.double().numpy()
        y_pred_np = y_pred

        eval_df = pd.DataFrame()
        eval_df['SMILES'] = test_smiles_arr
        for col_idx, col_name in enumerate(dataset.target_column_names):
            eval_df[col_name] = y_test_np[:, col_idx].tolist()
            eval_df[f"{col_name} (Prediction)"] = y_pred_np[:, col_idx].tolist()

        metrics_from_predictions("regression", eval_df, dataset.target_column_names)
        status = f"Evaluation completed."

        return f"<span style='color: green;'>{status}</span>", eval_df, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as exc:
        status = f"Error evaluating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, gr.update(interactive=False), gr.update(interactive=False)

def on_export_metrics(working_directory_path, file_name, eval_df, dataset):
    try:
        if dataset is None:
            raise ValueError("Please process the data and run an evaluation first.")
        metrics = metrics_from_predictions("regression", eval_df, dataset.target_column_names)
        file_name = export_result_csv(working_directory_path, file_name, metrics)
        status = f"Test metrics exported: {file_name}"
        return f"<span style='color: green;'>{escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting test metrics: {exc}"
        return f"<span style='color: red;'>{escape(status)}</span>", gr.update()

def on_export_evaluation(working_directory_path, file_name, eval_df):
    try:
        file_name = export_result_csv(working_directory_path, file_name, eval_df)
        status = f"Evaluation exported: {file_name}"
        return f"<span style='color: green;'>{escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting evaluation: {exc}"
        return f"<span style='color: red;'>{escape(status)}</span>", gr.update()

def on_process_prediction_data(working_directory_path, prediction_dataset_file, dataset, prediction_graph_directory, prediction_molecular_fingerprint_directory, datatype, batch_size):
    try:
        if dataset is None:
            raise ValueError("Please process the training data first.")
        # Process the prediction dataset and create a dataloader
        prediction_dataset_file = sanitize_path_component(prediction_dataset_file, "dataset file name")
        prediction_dataset_file_path = os.path.join(working_directory_path, prediction_dataset_file)
        prediction_graph_directory_path = (os.path.join(working_directory_path, sanitize_path_component(prediction_graph_directory, "graph directory"))
                                           if dataset.load_data in ('Graphs', 'Both') else None)
        prediction_molecular_fingerprint_directory_path = (os.path.join(working_directory_path, sanitize_path_component(prediction_molecular_fingerprint_directory, "molecular fingerprint directory"))
                                                           if dataset.load_data in ('Molecular fingerprints', 'Both') else None)
        prediction_dataset = MolecularDataset(prediction_dataset_file_path, dataset.load_data, prediction_graph_directory_path, prediction_molecular_fingerprint_directory_path, datatype, target_datatype=torch.float64)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size)
        
        status = "Prediction data processed successfully."
        return f"<span style='color: green;'>{status}</span>", prediction_dataset, prediction_dataloader, gr.update(interactive=True)
    except Exception as exc:
        status = f"Error processing prediction data: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, gr.update(interactive=False)

# Define the predict function
@torch.no_grad()
def predict(model, dataset, dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    model.eval()
    smiles_list = []
    y_pred_scaled_list = []
    need_edge_attr = model_requires_edge_attr(model)
    
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
                fingerprints_variance_thresholded = variance_threshold.transform(fingerprints.double().numpy())
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
        y_pred_scaled_list.append(output.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled_list, dim=0) if y_pred_scaled_list else torch.tensor([])
    y_pred = output_scaler.inverse_transform(y_pred_scaled.double().numpy()) if y_pred_scaled.numel() else np.empty((0, dataset.num_outputs))
    return smiles_list, y_pred

def on_predict(model, dataset, prediction_dataset, predict_dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca):
    try:
        if model is None or predict_dataloader is None:
            raise ValueError("Please process the prediction data first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data, or load a checkpoint that stores one.")
        ensure_model_on_device(model, device)
        # Get predictions
        smiles_list, y_pred = predict(model, prediction_dataset, predict_dataloader, device, datatype, output_scaler, gnn_model_tab, customgcn_convolutional_layer_name, variance_threshold, pca)
        if len(smiles_list) == 0:
            raise ValueError("The prediction dataset is empty.")
        y_pred_np = y_pred

        prediction_df = pd.DataFrame()
        prediction_df['SMILES'] = smiles_list
        for col_idx, col_name in enumerate(dataset.target_column_names):
            prediction_df[f"{col_name} (Prediction)"] = y_pred_np[:, col_idx].tolist()

        status = f"Prediction completed."

        return f"<span style='color: green;'>{status}</span>", prediction_df, gr.update(interactive=True)
    except Exception as exc:
        status = f"Error making prediction: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, gr.update(interactive=False)

def on_export_prediction(working_directory_path, prediction_file_name, prediction_df):
    try:
        if prediction_df is None:
            raise ValueError("Please make a prediction first.")
        filename = export_result_csv(working_directory_path, prediction_file_name, prediction_df)
        status = f"Predictions exported: {filename}"
        return f"<span style='color: green;'>{escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting predictions: {exc}"
        return f"<span style='color: red;'>{escape(status)}</span>", gr.update()

def regression_hybrid_tab_content():
    with gr.Tab("Regression - Hybrid GNN") as regression_tab:
        with gr.Row():
            with gr.Column(scale=1):
                working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(), value="wd", allow_custom_value=True,
                                                         info="Folders in Data/Regression. Select one or enter a new folder name.")
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
                with gr.Accordion("Upload Dataset"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Dataset file")
                            dataset_name_textbox = gr.Textbox(label="Dataset name", value="dataset")
                            smiles_column_textbox = gr.Textbox(label="SMILES column", value="SMILES")
                            target_columns_textbox = gr.Textbox(label="Target columns (separated by commas)", value="target")
                            canonicalize_smiles_checkbox, remove_duplicates_checkbox, remove_salt_checkbox = dataset_cleanup_controls()
                            upload_dataset_button = gr.Button(value="Load dataset", interactive=False)
                        with gr.Column(scale=2):
                            dataset_preview_dataframe = gr.Dataframe(label="Dataset preview", max_height=360, interactive=False)
                with gr.Accordion("Feature Extraction"):
                    with gr.Row():
                        dataset_file_feature_extraction_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Graph extraction**")
                            graph_directory_textbox = gr.Textbox(label="Graph directory", value="GRAPH_graphs", info="GRAPH_ is added automatically when omitted.")
                            graph_featurizer_dropdown = gr.Dropdown(label="Graph featurizer", value="MolGraphConvFeaturizer", choices=["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"])
                            extract_graphs_button = gr.Button(value="Extract graphs", interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Molecular fingerprint extraction**")
                            molecular_fingerprint_directory_textbox = gr.Textbox(label="Molecular fingerprint directory", value="FP_fingerprints", info="FP_ is added automatically when omitted.")
                            molecular_fingerprint_dropdown = gr.Dropdown(label="Molecular fingerprint", value="Morgan fingerprint", choices=["Morgan fingerprint", "Atom-pair fingerprint", "Topological-torsion fingerprint", "Avalon fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint"])
                            radius_slider = gr.Slider(label="Radius", value=2, minimum=1, maximum=5, step=1)
                            number_of_bits_slider = gr.Slider(label="Number of bits", value=2048, minimum=128, maximum=4096, step=128)
                            extract_molecule_fingerprints_button = gr.Button(value="Extract molecular fingerprints", interactive=False)
                with gr.Accordion("Load and process data"):
                    with gr.Row():
                        dataset_file_process_data_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                        load_data_radio = gr.Radio(label="Which data to load?", choices=["Graphs", "Molecular fingerprints", "Both"], value="Both")
                    with gr.Row():
                        with gr.Column(scale=1):
                            graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[], info="Extracted graphs in GRAPH_ folders.")
                            molecular_fingerprint_directory_dropdown = gr.Dropdown(label="Molecular fingerprint directory", choices=[], info="Extracted fingerprints in FP_ folders.")
                            dimensionality_reduction_checkbox = gr.Checkbox(label="Dimensionality reduction (for molecular fingerprints)", value=False)
                            variance_threshold_slider = gr.Slider(minimum=0, maximum=1.0, value=0.01, step=0.01, label="Variance threshold", visible=False)
                            pca_num_components_slider = gr.Slider(minimum=1, maximum=256, value=32, step=1, label="PCA number of components", visible=False)
                        with gr.Column(scale=1):
                            split_method_radio = gr.Radio(
                                label="Split method", choices=["Random", "Scaffold"], value="Random",
                                info="Scaffold keeps molecules with the same Murcko scaffold in one split. Sizes are approximate; all acyclic molecules share one scaffold.")
                            test_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Test size")
                            val_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Validation size")
                            data_split_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Data split random seed")
                            batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
                            shuffle_checkbox = gr.Checkbox(label="Shuffle training data each epoch", value=True,
                                                           info="Shuffling is seeded from the random seed, so runs stay reproducible. Uncheck for a fixed batch order.")
                            process_data_button = gr.Button(value="Process data", interactive=False)
                            dataset_state = gr.State()
                            train_dataloader_state = gr.State()
                            val_dataloader_state = gr.State()
                            test_dataloader_state = gr.State()
                            variance_threshold_state = gr.State()
                            pca_state = gr.State()
                            output_scaler_state = gr.State()
                    split_context_state = gr.State()
                    similarity_controls = build_similarity_controls()
                with gr.Accordion("Model"):
                    with gr.Row():
                        with gr.Column(scale=1) as graph_encoder_column:
                            gr.Markdown(value="**Graph Encoder**")
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
                                    customgcn_convolutional_layer_name = gr.Dropdown(label="Graph convolutional layer", value="GCNConv", choices=["GCNConv", "SAGEConv", "SGConv", "ClusterGCNConv", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "SuperGATConv", "TransformerConv", "GeneralConv"])
                                    customgcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                    customgcn_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                                    customgcn_n_heads = gr.Slider(label="n_heads", minimum=1, maximum=8, value=3, step=1)
                                    customgcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                    customgcn_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                        with gr.Column(scale=1) as fingerprint_encoder_column:
                            gr.Markdown(value="**Fingerprint Encoder**")
                            mlp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                            mlp_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                            mlp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                        with gr.Column(scale=1):
                            gr.Markdown(value="**Prediction Head**")
                            predictor_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                            predictor_n_layers = gr.Slider(label="n_layers", minimum=0, maximum=6, value=1, step=1)
                    with gr.Row():
                        model_name_textbox = gr.Textbox(label="Model name", value="regression_hybrid", info="Saved in MODEL_<model name>_<model seed>.")
                        model_random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Model random seed")
                    with gr.Row():
                        create_model_button = gr.Button(value="Create model", interactive=False)
                        run_directory_textbox = gr.Textbox(label="Run directory", interactive=False)
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
                            checkpoint_file_dropdown = gr.Dropdown(label="Checkpoint file", choices=[], info="Checkpoints inside MODEL_ folders.")
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
                            metrics_file_name_textbox = gr.Textbox(label="Metrics export file", value="test_metrics")
                            export_metrics_button = gr.Button(value="Export test metrics (all targets)", interactive=False)
                            evaluation_file_name_textbox = gr.Textbox(label="Evaluation export file", value="evaluation")
                            export_evaluation_button = gr.Button(value="Export evaluation rows (all targets)", interactive=False)
                            evaluation_df_state = gr.State()
                        with gr.Column(scale=2):
                            evaluation_plot = gr.Plot()
                with gr.Accordion("Make predictions"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_dataset_file_dropdown = gr.Dropdown(label="Prediction dataset file", choices=[])
                            prediction_graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[], info="Extracted graphs in GRAPH_ folders.")
                            prediction_molecular_fingerprint_directory_dropdown = gr.Dropdown(label="Molecular fingerprint directory", choices=[], info="Extracted fingerprints in FP_ folders.")
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
    concurrency_id = "regression_hybrid-" + str(split_context_state._id)
    directory_resets = directory_reset_updates(
        clear_states=[split_context_state, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state,
                      model_state, optimizer_state, schedular_state, evaluation_df_state, prediction_dataset_state,
                      prediction_dataloader_state, variance_threshold_state, pca_state, output_scaler_state],
        loss_states=[train_losses_state, val_losses_state], epoch_state=trained_epochs_state,
        disable_buttons=[create_model_button, create_optimizer_button, train_button, evaluate_button,
                         save_checkpoint_button, load_checkpoint_button, process_prediction_data_button,
                         predict_button, export_losses_button, export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[run_directory_textbox, dataset_preview_dataframe, training_plot, evaluation_metrics_dataframe, evaluation_plot,
                          prediction_dataframe, model_summary_textarea], similarity_controls=similarity_controls,
    )
    directory_resets[target_column_dropdown] = gr.update(choices=[], value=None)
    bind_working_directory(working_directory_dropdown, "change", on_open_working_directory, working_directory_dropdown,
                           [working_directory_dropdown, upload_dataset_button, extract_graphs_button, extract_molecule_fingerprints_button, process_data_button, working_directory_path_state, working_directory_content_list_state],
                           reset_updates=directory_resets, concurrency_id=concurrency_id)
    bind_working_directory(open_working_directory_button, "click", on_open_working_directory, working_directory_dropdown,
                           [working_directory_dropdown, upload_dataset_button, extract_graphs_button, extract_molecule_fingerprints_button, process_data_button, working_directory_path_state, working_directory_content_list_state],
                           reset_updates=directory_resets, concurrency_id=concurrency_id)
    bind_similarity_controls(similarity_controls, split_context_state, status_html, concurrency_id=concurrency_id)
    working_directory_content_list_state.change(on_content_list_change, [working_directory_path_state, dataset_name_textbox, graph_directory_textbox, molecular_fingerprint_directory_textbox, checkpoint_name_textbox],
                                                [working_directory_content_list_dataframe,
                                                 dataset_file_feature_extraction_dropdown, graph_directory_dropdown, molecular_fingerprint_directory_dropdown, dataset_file_process_data_dropdown, checkpoint_file_dropdown, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown])
    
    # Settings interaction
    datatype_dropdown.change(on_datatype_change, datatype_dropdown, datatype_state)

    # Load dataset and process data interactions
    bind_artifact_update(upload_dataset_button, on_upload_dataset, [dataset_file, working_directory_path_state, dataset_name_textbox, smiles_column_textbox, target_columns_textbox,
                                                  canonicalize_smiles_checkbox, remove_duplicates_checkbox, remove_salt_checkbox],
                                [status_html, dataset_preview_dataframe, working_directory_content_list_state], dataset_state=dataset_state, reset_updates=directory_resets, kind="upload", concurrency_id=concurrency_id)

    # Feature extraction interactions
    bind_artifact_update(extract_graphs_button, on_extract_graphs, [working_directory_path_state, dataset_file_feature_extraction_dropdown, graph_directory_textbox, graph_featurizer_dropdown, datatype_state], [status_html, working_directory_content_list_state], dataset_state=dataset_state, reset_updates=directory_resets, kind="graphs", concurrency_id=concurrency_id)
    molecular_fingerprint_dropdown.change(on_molecular_fingerprint_change, molecular_fingerprint_dropdown, radius_slider)
    load_data_radio.change(on_load_data_option_change, [load_data_radio, dimensionality_reduction_checkbox], [graph_directory_dropdown, molecular_fingerprint_directory_dropdown, dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider, graph_encoder_column, fingerprint_encoder_column, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown])
    bind_artifact_update(extract_molecule_fingerprints_button, on_extract_molecule_fingerprints, [working_directory_path_state, dataset_file_feature_extraction_dropdown, molecular_fingerprint_directory_textbox, molecular_fingerprint_dropdown, radius_slider, number_of_bits_slider], [status_html, working_directory_content_list_state], dataset_state=dataset_state, reset_updates=directory_resets, kind="fingerprints", concurrency_id=concurrency_id)

    # Process data interactions
    dimensionality_reduction_checkbox.change(on_dimensionality_reduction_change, dimensionality_reduction_checkbox, [variance_threshold_slider, pca_num_components_slider])
    bind_data_processing(
        process_data_button, capture_split_outputs(on_process_data, 3, similarity_controls),
        [working_directory_path_state, dataset_file_process_data_dropdown, load_data_radio, graph_directory_dropdown, molecular_fingerprint_directory_dropdown, datatype_state, dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider, test_size_slider, val_size_slider, batch_size_dropdown, shuffle_checkbox, data_split_seed_slider, split_method_radio],
        [status_html, create_model_button, target_column_dropdown, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state, variance_threshold_state, pca_state, output_scaler_state, split_context_state, *similarity_controls.reset_outputs],
        clear_states=[model_state, optimizer_state, schedular_state, evaluation_df_state,
                      prediction_dataset_state, prediction_dataloader_state],
        loss_states=[train_losses_state, val_losses_state], epoch_state=trained_epochs_state, concurrency_id=concurrency_id,
        disable_buttons=[create_optimizer_button, train_button, evaluate_button,
                         save_checkpoint_button, load_checkpoint_button, process_prediction_data_button,
                         predict_button, export_losses_button, export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[run_directory_textbox, model_summary_textarea, training_plot, evaluation_metrics_dataframe,
                          evaluation_plot, prediction_dataframe],
    )
    
    # Create model interactions
    result_reset = dict(
        concurrency_id=concurrency_id,
        clear_states=[evaluation_df_state],
        disable_buttons=[export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[evaluation_metrics_dataframe, evaluation_plot, prediction_dataframe],
    )
    bind_result_invalidation(create_model_button, show_model_run(on_create_model, 8), [dataset_state, gnn_model_tab_state, device_dropdown, datatype_state, model_random_seed_slider, variance_threshold_state, pca_state,
                                                gcn_n_hiddens, gcn_num_layers, gcn_dropout, gcn_n_outputs,
                                                graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_dropout, graph_sage_n_outputs,
                                                gin_n_hiddens, gin_num_layers, gin_dropout, gin_n_outputs,
                                                gat_n_hiddens, gat_num_layers, gat_dropout, gat_n_outputs,
                                                edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_dropout, edgecnn_n_outputs,
                                                attentivefp_n_hiddens, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_dropout, attentivefp_n_outputs,
                                                customgcn_convolutional_layer_name, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                                mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                predictor_n_hiddens, predictor_n_layers, model_name_textbox],
                                                [status_html, model_summary_textarea, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, process_prediction_data_button, train_button, model_state, train_losses_state, val_losses_state, trained_epochs_state, run_directory_textbox], **result_reset)
    save_checkpoint_button.click(on_save_checkpoint, [working_directory_path_state, checkpoint_name_textbox, model_state, trained_epochs_state, variance_threshold_state, pca_state, output_scaler_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    bind_result_invalidation(load_checkpoint_button, show_model_run(on_load_checkpoint, 1, restore_controls=True), [working_directory_path_state, checkpoint_file_dropdown, model_state, train_losses_state, val_losses_state, trained_epochs_state, variance_threshold_state, pca_state, output_scaler_state],
                                 [status_html, model_state, train_losses_state, val_losses_state, trained_epochs_state, variance_threshold_state, pca_state, output_scaler_state, run_directory_textbox, model_name_textbox, model_random_seed_slider], **result_reset)
    
    # Train model interactions
    create_optimizer_button.click(on_create_optimizer, [model_state, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [status_html, optimizer_state, schedular_state, train_button], concurrency_id=concurrency_id)
    bind_result_invalidation(train_button, on_train, [model_state, dataset_state, train_dataloader_state, val_dataloader_state, device_dropdown, datatype_state, output_scaler_state,
                                  optimizer_state, schedular_state, epochs_slider, trained_epochs_state, train_losses_state, val_losses_state,
                                  gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                  [status_html, training_plot, export_losses_button, trained_epochs_state, train_losses_state, val_losses_state], **result_reset)
    export_losses_button.click(route_model_export(on_export_losses), [working_directory_path_state, loss_file_name_textbox, train_losses_state, val_losses_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    
    # Evaluate model interactions
    evaluate_button.click(on_evaluate, [model_state, dataset_state, test_dataloader_state, device_dropdown, datatype_state, output_scaler_state,
                                        gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                        [status_html, evaluation_df_state, export_evaluation_button, export_metrics_button], concurrency_id=concurrency_id
                         ).then(on_target_column_change, [evaluation_df_state, target_column_dropdown], [evaluation_metrics_dataframe, evaluation_plot], concurrency_id=concurrency_id)
    target_column_dropdown.change(on_target_column_change, [evaluation_df_state, target_column_dropdown], [evaluation_metrics_dataframe, evaluation_plot], concurrency_id=concurrency_id)
    export_evaluation_button.click(route_model_export(on_export_evaluation), [working_directory_path_state, evaluation_file_name_textbox, evaluation_df_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    export_metrics_button.click(route_model_export(on_export_metrics), [working_directory_path_state, metrics_file_name_textbox, evaluation_df_state, dataset_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
        
    # Make prediction interactions
    process_prediction_data_button.click(on_process_prediction_data, [working_directory_path_state, prediction_dataset_file_dropdown, dataset_state, prediction_graph_directory_dropdown, prediction_molecular_fingerprint_directory_dropdown, datatype_state, batch_size_dropdown], [status_html, prediction_dataset_state, prediction_dataloader_state, predict_button], concurrency_id=concurrency_id)
    predict_button.click(on_predict, [model_state, dataset_state, prediction_dataset_state, prediction_dataloader_state, device_dropdown, datatype_state, output_scaler_state,
                                      gnn_model_tab_state, customgcn_convolutional_layer_name, variance_threshold_state, pca_state],
                                      [status_html, prediction_dataframe, export_prediction_button], concurrency_id=concurrency_id)
    export_prediction_button.click(route_model_export(on_export_prediction), [working_directory_path_state, prediction_file_name_textbox, prediction_dataframe, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    
    return regression_tab
