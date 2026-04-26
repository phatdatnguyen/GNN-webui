import math
import os
import time
import pandas as pd
import gradio as gr
import torch
from torch_geometric.nn import summary
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus, ViSNet
from torch_geometric.loader import DataLoader
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from datasets import MolecularDataset3D
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
        return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), None, None
    
    working_directory_path = os.path.join("./Data/", working_directory_name)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_working_directory_contents(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory_name), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), working_directory_path, files

def on_content_list_change(working_directory_path, dataset_file_name, graph_directory, checkpoint_file_name):
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

    return contents_df, \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=directories, value=graph_directory_value), \
           gr.update(choices=data_files, value=dataset_file_name_value), \
           gr.update(choices=checkpoint_files, value=checkpoint_file_name_value), \
           gr.update(choices=data_files, value=prediction_dataset_file_name_value), \
           gr.update(choices=directories, value=prediction_graph_directory_value)

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

    status, df = process_dataset_file(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns, True)
    preview = df.head(10) if df is not None else None
    return status, preview, get_working_directory_contents(working_directory_path)

def on_extract_graphs(working_directory_path, dataset_file_name, graph_directory, number_of_conformers, force_field, datatype, progress=gr.Progress()):
    status = extract_3d_graphs(working_directory_path, dataset_file_name, graph_directory, number_of_conformers, force_field, datatype, progress)
    return status, get_working_directory_contents(working_directory_path)

def on_process_data(working_directory_path, dataset_file_name, graph_directory, datatype, test_ratio, val_ratio, batch_size, random_seed):
    try:
        # Validate inputs
        if graph_directory is None or graph_directory.strip() == "":
            raise ValueError("Please select a graph directory.")

        # Validate output columns
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        target_columns_list = df.drop(columns=['SMILES']).columns.tolist()
        if len(target_columns_list) == 0:
            raise ValueError("No target columns found in the dataset. Please ensure your dataset has at least one target column in addition to the SMILES column.")
        for col in target_columns_list:
            if not pd.api.types.is_numeric_dtype(df[col]) :
                raise ValueError(f"Target column '{col}' is not numeric. Please ensure all target columns are numeric.")

        # Set up dataset and dataloaders
        graph_directory_path = os.path.join(working_directory_path, graph_directory)

        dataset = MolecularDataset3D(dataset_file_path, graph_directory_path, datatype)
        
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
        
        # Set up output scaler (fit on training split only to avoid leakage)
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        train_targets = dataset.df.drop(columns=['SMILES']).iloc[train_dataset.indices].to_numpy()
        output_scaler.fit(train_targets)

        status = f"Data processed: {total_len} molecules."
        return f"<span style='color: green;'>{status}</span>", gr.update(interactive=True), dataset, train_dataloader, val_dataloader, test_dataloader, output_scaler
    except Exception as exc:
        status = f"Error processing data: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, None, None, None, None

def on_gnn_model_tab_selected(evt: gr.SelectData):
    return evt.value

def on_create_model(dataset, gnn_model_tab, device, datatype, random_seed,
                    schnet_n_hiddens, schnet_n_filters, schnet_n_interactions, schnet_n_gaussians, schnet_cutoff, schnet_max_num_neighbors, schnet_readout, schnet_dipole,
                    dimenetplusplus_n_hiddens, dimenetplusplus_n_blocks, dimenetplusplus_int_emb_size, dimenetplusplus_basis_emb_size, dimenetplusplus_out_emb_channels, dimenetplusplus_n_spherical, dimenetplusplus_n_radial, dimenetplusplus_cutoff, dimenetplusplus_max_num_neighbors, dimenetplusplus_envelope_exponent, dimenetplusplus_n_output_layers,
                    visnet_n_hiddens, visnet_n_heads, visnet_n_layers, visnet_num_rbf, visnet_trainable_rbf, visnet_max_z, visnet_cutoff, visnet_max_num_neighbors, visnet_vertex):
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)

    model = None
    train_losses = []
    val_losses = []
    trained_epochs = 0

    try:
        model = None
        if gnn_model_tab == "SchNet":
            model = SchNet(
                hidden_channels=schnet_n_hiddens,
                num_filters=schnet_n_filters,
                num_interactions=schnet_n_interactions,
                num_gaussians=schnet_n_gaussians,
                cutoff=schnet_cutoff,
                max_num_neighbors=schnet_max_num_neighbors,
                readout=schnet_readout,
                dipole=schnet_dipole
            ).to(device=device, dtype=datatype)
        elif gnn_model_tab == "DimeNet++":
            model = DimeNetPlusPlus(
                hidden_channels=dimenetplusplus_n_hiddens,
                out_channels=1,
                num_blocks=dimenetplusplus_n_blocks,
                int_emb_size=dimenetplusplus_int_emb_size,
                basis_emb_size=dimenetplusplus_basis_emb_size,
                out_emb_channels=dimenetplusplus_out_emb_channels,
                num_spherical=dimenetplusplus_n_spherical,
                num_radial=dimenetplusplus_n_radial,
                cutoff=dimenetplusplus_cutoff,
                max_num_neighbors=dimenetplusplus_max_num_neighbors,
                envelope_exponent=dimenetplusplus_envelope_exponent,
                num_output_layers=dimenetplusplus_n_output_layers
            ).to(device=device, dtype=datatype)
        elif gnn_model_tab == "ViSNet":
            model = ViSNet(
                num_heads=visnet_n_heads,
                num_layers=visnet_n_layers,
                hidden_channels=visnet_n_hiddens,
                num_rbf=visnet_num_rbf,
                trainable_rbf=visnet_trainable_rbf,
                max_z=visnet_max_z,
                cutoff=visnet_cutoff,
                max_num_neighbors=visnet_max_num_neighbors,
                vertex=visnet_vertex
            ).to(device=device, dtype=datatype)
        
        # Reset training state
        train_losses = []
        val_losses = []
        trained_epochs = 0
        
        status = "Model created successfully."
        return f"<span style='color: green;'>{status}</span>", \
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), \
            model, train_losses, val_losses, trained_epochs
    except Exception as exc:
        status = f"Error creating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", \
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

loss_fn = torch.nn.MSELoss()
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
def train(model, dataloader, device, datatype, output_scaler, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0
        
    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.int32)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y), device=device, dtype=datatype)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(z, pos, batch_index)
        if isinstance(output, tuple):
            output = output[0]

        # Compute loss and backpropagate
        loss = loss_fn(output, y_scaled)
        loss.backward()
        optimizer.step()
        total_loss += torch.sqrt(loss).item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

# Define the validation function
@torch.no_grad()
def validation(model, dataloader, device, datatype, output_scaler):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.int32)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y), device=device, dtype=datatype)
        
        with torch.no_grad():
            # Forward pass
            output = model(z, pos, batch_index)
            if isinstance(output, tuple):
                output = output[0]

            # Compute loss
            loss = loss_fn(output, y_scaled)
            total_loss += torch.sqrt(loss).item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0

def on_train(model, train_dataloader, val_dataloader, device, datatype, output_scaler, optimizer, scheduler, epochs, trained_epochs, train_losses, val_losses, progress=gr.Progress()):
    # Training loop
    for _ in progress.tqdm(range(trained_epochs, epochs + trained_epochs), total=epochs, desc="Training"):
        train_loss = train(model, train_dataloader, device, datatype, output_scaler, optimizer)
        train_losses.append(train_loss)
        val_loss = validation(model, val_dataloader, device, datatype, output_scaler)
        val_losses.append(val_loss)
        scheduler.step()
        trained_epochs += 1

    status = f"Training completed for {trained_epochs} epochs."

    # Plot the training and validation loss
    figure = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE loss')
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
def test(model, dataloader, device, datatype, output_scaler):
    model.eval()
    y_test_list = []
    y_pred_scaled_list = []
    smiles_arr = []
    
    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.int32)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y = batch.y.to(dtype=datatype)
        smiles = batch.smiles

        with torch.no_grad():
            # Forward pass
            output = model(z, pos, batch_index)
            if isinstance(output, tuple):
                output = output[0]

        smiles_arr.extend(smiles)
        y_test_list.append(y.detach())
        y_pred_scaled_list.append(output.cpu().detach())

    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.tensor([])
    y_pred_scaled = torch.cat(y_pred_scaled_list, dim=0) if y_pred_scaled_list else torch.tensor([])
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    return y_test, y_pred, smiles_arr

def on_evaluate(model, dataset, test_dataloader, device, datatype, output_scaler):
    # Get test predictions
    y_test, y_pred, test_smiles_arr = test(model, test_dataloader, device, datatype, output_scaler)
    y_test_np = y_test.detach().numpy().ravel()
    y_pred_np = y_pred.ravel()

    eval_df = pd.DataFrame()
    eval_df['SMILES'] = test_smiles_arr
    target_column = dataset.target_column_name
    eval_df[target_column] = y_test_np.tolist()
    eval_df[f"{target_column} (Prediction)"] = y_pred_np.tolist()

    # Evaluate the model using different metrics
    mae = round(mean_absolute_error(y_test_np, y_pred_np), 4)
    mse = round(mean_squared_error(y_test_np, y_pred_np), 4)
    rmse = round(math.sqrt(mse), 4)
    r2 = round(r2_score(y_test_np, y_pred_np), 4)

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
        'Value': [mae, mse, rmse, r2]
    })

    # Visualization
    scatter_plot = plt.figure(figsize=(8, 8))
    plt.scatter(y_test_np, y_pred_np)
    plt.xlabel('actual value')
    plt.ylabel('predicted value')
    plt.grid(True)
    plt.close()

    status = f"Evaluation completed."

    return f"<span style='color: green;'>{status}</span>", eval_df, metrics_df, scatter_plot, gr.update(interactive=True)

def on_export_evaluation(working_directory_path, file_name, eval_df):
    file_path = os.path.join(working_directory_path, f"{file_name}.csv")
    eval_df.to_csv(file_path)
    status = f"Evaluation exported: {file_name}.csv"

    return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)

def on_process_prediction_data(working_directory_path, prediction_dataset_file, prediction_graph_directory, datatype, batch_size):
    try:
        # Process the prediction dataset and create a dataloader
        prediction_dataset_file_path = os.path.join(working_directory_path, prediction_dataset_file)
        prediction_graph_directory_path = os.path.join(working_directory_path, prediction_graph_directory)                                        
        prediction_dataset = MolecularDataset3D(prediction_dataset_file_path, prediction_graph_directory_path, datatype)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size)
        
        status = "Prediction data processed successfully."
        return f"<span style='color: green;'>{status}</span>", prediction_dataset, prediction_dataloader, gr.update(interactive=True)
    except Exception as exc:
        status = f"Error processing prediction data: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, gr.update(interactive=False)

# Define the predict function
@torch.no_grad()
def predict(model, dataloader, device, datatype, output_scaler):
    model.eval()
    smiles_list = []
    y_pred_scaled_list = []
    
    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.int32)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        with torch.no_grad():
            # Forward pass
            output = model(z, pos, batch_index)
            if isinstance(output, tuple):
                output = output[0]
            
        smiles_list.extend(batch.smiles)
        y_pred_scaled_list.append(output.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled_list, dim=0) if y_pred_scaled_list else torch.tensor([])
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    return smiles_list, y_pred

def on_predict(model, dataset, predict_dataloader, device, datatype, output_scaler):
    # Get predictions
    smiles_list, y_pred = predict(model, predict_dataloader, device, datatype, output_scaler)
    y_pred_np = y_pred.ravel()

    prediction_df = pd.DataFrame()
    prediction_df['SMILES'] = smiles_list
    target_column = dataset.target_column_name
    prediction_df[f"{target_column} (Prediction)"] = y_pred_np.tolist()

    status = f"Prediction completed."

    return f"<span style='color: green;'>{status}</span>", prediction_df, gr.update(interactive=True)

def on_export_prediction(working_directory_path, prediction_file_name, prediction_df):
    prediction_file_path = os.path.join(working_directory_path, f"{prediction_file_name}.csv")
    prediction_df.to_csv(prediction_file_path)
    status = f"Predictions exported: {prediction_file_name}.csv"
    return f"<span style='color: green;'>{status}</span>", get_working_directory_contents(working_directory_path)

def regression_3d_tab_content():
    with gr.Tab("Regression 3D") as regression_3d_tab:
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
                with gr.Accordion("3D Graph Extraction"):
                    with gr.Row():
                        dataset_file_feature_extraction_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                        graph_directory_textbox = gr.Textbox(label="Graph directory", value="3d_graphs")
                    with gr.Row():
                        number_of_conformers_slider = gr.Slider(label="Number of conformers", minimum=1, maximum=100, value=30, step=1)
                        force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], allow_custom_value=False)
                        extract_graphs_button = gr.Button(value="Extract graphs", interactive=False)
                with gr.Accordion("Load and process data"):
                    with gr.Row():
                        dataset_file_process_data_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[])
                        with gr.Column(scale=1):
                            test_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Test size")
                            val_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Validation size")
                            batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
                            process_data_button = gr.Button(value="Process data", interactive=False)
                            dataset_state = gr.State()
                            train_dataloader_state = gr.State()
                            val_dataloader_state = gr.State()
                            test_dataloader_state = gr.State()
                            output_scaler_state = gr.State()
                with gr.Accordion("3D GNN model"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gnn_model_tab_state = gr.State(value="SchNet")
                            with gr.Tabs():
                                with gr.Tab("SchNet") as schnet_tab:
                                    with gr.Row(equal_height=True):
                                        with gr.Column(scale=1):
                                            schnet_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                            schnet_n_filters = gr.Dropdown(label="n_filters", value=128, choices=[16, 32, 64, 128, 256, 512])
                                            schnet_n_interactions = gr.Slider(label="n_interactions", minimum=1, maximum=12, value=6, step=1)
                                            schnet_n_gaussians = gr.Slider(label="n_gaussians", minimum=10, maximum=100, value=50, step=1)
                                            schnet_cutoff = gr.Slider(label="cutoff", minimum=5.0, maximum=20.0, value=10.0, step=0.1)
                                            schnet_max_num_neighbors = gr.Slider(label="max_num_neighbors", minimum=1, maximum=64, value=32, step=1)
                                            schnet_readout = gr.Radio(label="readout", value="add", choices=["add", "mean"])
                                            schnet_dipole = gr.Checkbox(label="dipole", value=False)
                                    schnet_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("DimeNet++") as dimenetplusplus_tab:
                                    with gr.Row(equal_height=True):
                                        with gr.Column(scale=1):
                                            dimenetplusplus_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                            dimenetplusplus_n_blocks = gr.Slider(label="n_blocks", minimum=1, maximum=12, value=6, step=1)
                                            dimenetplusplus_int_emb_size = gr.Dropdown(label="int_emb_size", value=8, choices=[2, 4, 8, 16, 32, 64])
                                            dimenetplusplus_basis_emb_size = gr.Dropdown(label="basis_emb_size", value=8, choices=[2, 4, 8, 16, 32, 64])
                                            dimenetplusplus_out_emb_channels = gr.Dropdown(label="out_emb_channels", value=8, choices=[2, 4, 8, 16, 32, 64])
                                            dimenetplusplus_n_spherical = gr.Slider(label="n_spherical", minimum=1, maximum=12, value=7, step=1)
                                            dimenetplusplus_n_radial = gr.Slider(label="n_radial", minimum=1, maximum=12, value=6, step=1)
                                            dimenetplusplus_cutoff = gr.Slider(label="cutoff", minimum=5.0, maximum=20.0, value=10.0, step=0.1)
                                            dimenetplusplus_max_num_neighbors = gr.Slider(label="max_num_neighbors", minimum=1, maximum=64, value=32, step=1)
                                            dimenetplusplus_envelope_exponent = gr.Slider(label="envelope_exponent", minimum=1, maximum=10, value=5, step=1)
                                            dimenetplusplus_n_output_layers = gr.Slider(label="n_output_layers", minimum=1, maximum=6, value=3, step=1)
                                    dimenetplusplus_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                                with gr.Tab("ViSNet") as visnet_tab:
                                    with gr.Row(equal_height=True):
                                        with gr.Column(scale=1):
                                            visnet_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                            visnet_n_heads = gr.Slider(label="n_heads", minimum=1, maximum=32, value=8, step=1)
                                            visnet_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=12, value=6, step=1)
                                            visnet_num_rbf = gr.Slider(label="num_rbf", minimum=1, maximum=64, value=32, step=1)
                                            visnet_trainable_rbf = gr.Checkbox(label="trainable_rbf", value=False)
                                            visnet_max_z = gr.Slider(label="max_z", minimum=1, maximum=500, value=100, step=1)
                                            visnet_cutoff = gr.Slider(label="cutoff", minimum=5.0, maximum=20.0, value=10.0, step=0.1)
                                            visnet_max_num_neighbors = gr.Slider(label="max_num_neighbors", minimum=1, maximum=64, value=32, step=1)
                                            visnet_vertex = gr.Checkbox(label="vertex", value=False)
                                    visnet_tab.select(on_gnn_model_tab_selected, [], gnn_model_tab_state)
                        with gr.Column(scale=1):
                            create_model_button = gr.Button(value="Create model", interactive=False)
                            model_state = gr.State()
                            train_losses_state = gr.State()
                            val_losses_state = gr.State()
                            trained_epochs_state = gr.State()
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
                                learning_rate_slider = gr.Slider(label="Learning rate", minimum=0.000001, maximum=0.1, value=0.00001, step=0.000001)
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
                            evaluation_metrics_dataframe = gr.DataFrame(label="Evaluation metrics", wrap=False, interactive=False)
                            evaluation_file_name_textbox = gr.Textbox(label="File name", value="evaluation")
                            export_evaluation_button = gr.Button(value="Export", interactive=False)
                            evaluation_df_state = gr.State()
                        with gr.Column(scale=1):
                            evaluation_plot = gr.Plot()
                with gr.Accordion("Make prediction"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_dataset_file_dropdown = gr.Dropdown(label="Prediction dataset file", choices=[])
                            prediction_graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[])
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
    working_directory_dropdown.change(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, upload_dataset_button, extract_graphs_button, process_data_button, working_directory_path_state, working_directory_content_list_state])
    open_working_directory_button.click(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, upload_dataset_button, extract_graphs_button, process_data_button, working_directory_path_state, working_directory_content_list_state])
    working_directory_content_list_state.change(on_content_list_change, [working_directory_path_state, dataset_name_textbox, graph_directory_textbox, checkpoint_name_textbox],
                                                [working_directory_content_list_dataframe,
                                                 dataset_file_feature_extraction_dropdown, graph_directory_dropdown, dataset_file_process_data_dropdown, checkpoint_file_dropdown, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown])
    
    # Settings interaction
    datatype_dropdown.change(on_datatype_change, datatype_dropdown, datatype_state)

    # Load dataset and process data interactions
    upload_dataset_button.click(on_upload_dataset, [dataset_file, working_directory_path_state, dataset_name_textbox, smiles_column_textbox, target_columns_textbox], [status_html, dataset_preview_dataframe, working_directory_content_list_state])

    # Feature extraction interactions
    extract_graphs_button.click(on_extract_graphs, [working_directory_path_state, dataset_file_feature_extraction_dropdown, graph_directory_textbox, number_of_conformers_slider, force_field_dropdown, datatype_state], [status_html, working_directory_content_list_state])
    
    # Process data interactions
    process_data_button.click(on_process_data, [working_directory_path_state, dataset_file_process_data_dropdown, graph_directory_dropdown, datatype_state, test_size_slider, val_size_slider, batch_size_dropdown, random_seed_slider],
                              [status_html, create_model_button, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state, output_scaler_state])
    
    # Create model interactions
    create_model_button.click(on_create_model, [dataset_state, gnn_model_tab_state, device_dropdown, datatype_state, random_seed_slider,
                                                schnet_n_hiddens, schnet_n_filters, schnet_n_interactions, schnet_n_gaussians, schnet_cutoff, schnet_max_num_neighbors, schnet_readout, schnet_dipole,
                                                dimenetplusplus_n_hiddens, dimenetplusplus_n_blocks, dimenetplusplus_int_emb_size, dimenetplusplus_basis_emb_size, dimenetplusplus_out_emb_channels, dimenetplusplus_n_spherical, dimenetplusplus_n_radial, dimenetplusplus_cutoff, dimenetplusplus_max_num_neighbors, dimenetplusplus_envelope_exponent, dimenetplusplus_n_output_layers,
                                                visnet_n_hiddens, visnet_n_heads, visnet_n_layers, visnet_num_rbf, visnet_trainable_rbf, visnet_max_z, visnet_cutoff, visnet_max_num_neighbors, visnet_vertex],
                                                [status_html, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, process_prediction_data_button, model_state, train_losses_state, val_losses_state, trained_epochs_state])
    save_checkpoint_button.click(on_save_checkpoint, [working_directory_path_state, checkpoint_name_textbox, model_state, trained_epochs_state], [status_html, working_directory_content_list_state])
    load_checkpoint_button.click(on_load_checkpoint, [working_directory_path_state, checkpoint_file_dropdown, model_state], [status_html, model_state, train_losses_state, val_losses_state, trained_epochs_state])
    
    # Train model interactions
    create_optimizer_button.click(on_create_optimizer, [model_state, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [status_html, optimizer_state, schedular_state, train_button])
    train_button.click(on_train, [model_state, train_dataloader_state, val_dataloader_state, device_dropdown, datatype_state, output_scaler_state,
                                  optimizer_state, schedular_state, epochs_slider, trained_epochs_state, train_losses_state, val_losses_state],
                                 [status_html, training_plot, export_losses_button, trained_epochs_state, train_losses_state, val_losses_state])
    export_losses_button.click(on_export_losses, [working_directory_path_state, loss_file_name_textbox, train_losses_state, val_losses_state], [status_html, working_directory_content_list_state])
    
    # Evaluate model interactions
    evaluate_button.click(on_evaluate, [model_state, dataset_state, test_dataloader_state, device_dropdown, datatype_state, output_scaler_state],
                                       [status_html, evaluation_df_state, evaluation_metrics_dataframe, evaluation_plot, export_evaluation_button])
    export_evaluation_button.click(on_export_evaluation, [working_directory_path_state, evaluation_file_name_textbox, evaluation_df_state], [status_html, working_directory_content_list_state])
        
    # Make prediction interactions
    process_prediction_data_button.click(on_process_prediction_data, [working_directory_path_state, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown, datatype_state, batch_size_dropdown], [status_html, prediction_dataset_state, prediction_dataloader_state, predict_button])
    predict_button.click(on_predict, [model_state, dataset_state, prediction_dataloader_state, device_dropdown, datatype_state, output_scaler_state],
                                     [status_html, prediction_dataframe, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [working_directory_path_state, prediction_file_name_textbox, prediction_dataframe], [status_html, working_directory_content_list_state])
    
    return regression_3d_tab