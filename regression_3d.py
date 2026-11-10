import html
import math
import os
import time
import pandas as pd
import gradio as gr
import torch
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus, ViSNet
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from evaluation_metrics import metrics_from_predictions
from matplotlib import pyplot as plt
from datasets import MolecularDataset3D
from ui_utils import bind_data_processing, bind_result_invalidation
from update_dataset import build_dataset_upload_controls, upload_dataset
import feature_extraction
from gnn_run_ui import attach_model_run, capture_split_outputs, route_model_export, bind_working_directory, load_run_checkpoint, ensure_model_split, directory_reset_updates, show_model_run, capture_feature_sources, bind_artifact_update, bind_checkpoint_save, reset_checkpoint_optimizer, loss_epoch_numbers
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
        working_directory_path = os.path.join(DATA_DIR, "Regression", working_directory_name)
        os.makedirs(working_directory_path, exist_ok=True)
        files = get_working_directory_contents(working_directory_path)
        choices = get_working_directories()
    except (ValueError, OSError) as exc:
        gr.Warning(str(exc))
        return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), None, None

    return gr.update(choices=choices, value=working_directory_name), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), working_directory_path, files

def on_content_list_change(working_directory_path, dataset_file_name, graph_directory, checkpoint_file_name):
    if not working_directory_path:
        # Reachable when opening a working directory failed and the path state was cleared.
        empty = gr.update(choices=[], value=None)
        return pd.DataFrame(columns=["File", "Type", "Modified"]), empty, empty, empty, empty, empty, empty

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
        try:
            modified_timestamp = os.path.getmtime(content_path)
        except OSError:
            # Entries may disappear between listing and stat, including broken links.
            continue
        contents_info.append([content, file_type, time.ctime(modified_timestamp), modified_timestamp])
    # Sort on the numeric timestamp: the ctime string sorts by weekday name, not by date.
    contents_info.sort(key=lambda x: x[3], reverse=True)
    contents_df = pd.DataFrame([row[:3] for row in contents_info], columns=["File", "Type", "Modified"])

    # Filter files
    data_files = sorted(f for f in contents if f.endswith('.csv')
                        and os.path.isfile(os.path.join(working_directory_path, f)))
    try:
        directories = list_feature_directories(working_directory_path, "graph")
    except OSError:
        directories = []
    try:
        graph_directory = prefixed_feature_directory(graph_directory, "graph")
    except ValueError:
        graph_directory = None
    try:
        checkpoint_files = list_saved_models(working_directory_path, ".ckpt")
    except OSError:
        checkpoint_files = []

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

def on_upload_dataset(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns,
                      canonicalize_smiles=True, remove_duplicates=True, remove_salt=True):
    return upload_dataset(
        dataset_file_path, working_directory_path, dataset_name, smiles_column, target_columns,
        canonicalize_smiles, remove_duplicates, remove_salt, is_3d_dataset=True)

def on_extract_graphs(working_directory_path, dataset_file_name, graph_directory, number_of_conformers, force_field, datatype, progress=gr.Progress()):
    return feature_extraction.on_extract_3d_graphs(
        working_directory_path, dataset_file_name, graph_directory, number_of_conformers, force_field,
        datatype, progress)

def on_process_data(working_directory_path, dataset_file_name, graph_directory, datatype, test_ratio, val_ratio, batch_size, shuffle_training_data, random_seed, split_method="Random"):
    try:
        # Validate inputs
        random_seed = validate_seed(random_seed)
        if graph_directory is None or graph_directory.strip() == "":
            raise ValueError("Please select a graph directory.")

        # Validate output columns
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        target_columns_list = df.drop(columns=['SMILES']).columns.tolist()
        if len(target_columns_list) == 0:
            raise ValueError("No target columns found in the dataset. Please ensure your dataset has at least one target column in addition to the SMILES column.")
        for col in target_columns_list:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Target column '{col}' is not numeric. Please ensure all target columns are numeric.")
            if not np.isfinite(df[col].to_numpy()).all():
                raise ValueError(f"Target column '{col}' contains missing or non-finite values. Please remove or replace them before training.")

        # Set up dataset and dataloaders
        graph_directory_path = os.path.join(working_directory_path, graph_directory)

        # Preserve raw target precision until normalization, including low-precision runs.
        dataset = MolecularDataset3D(dataset_file_path, graph_directory_path, datatype, target_datatype=torch.float64)
        
        total_len = len(dataset)
        capture_feature_sources(dataset, graph_directory_path)
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
        
        # Set up output scaler (fit on training split only to avoid leakage)
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        train_targets = dataset.df.drop(columns=['SMILES']).iloc[train_dataset.indices].to_numpy()
        output_scaler.fit(train_targets)

        status = (f"Data processed: {total_len} molecules. {split_method} split: "
                  f"{len(train_dataset)} training, {len(val_dataset)} validation, "
                  f"{len(test_dataset)} test.")
        if split_method == "Scaffold":
            status += " Scaffold groups are kept together, so split sizes may differ from the requested ratios."
        status += " Create a model for this split before training or evaluation."
        return f"<span style='color: green;'>{status}</span>", gr.update(interactive=True), dataset, train_dataloader, val_dataloader, test_dataloader, output_scaler
    except Exception as exc:
        status = f"Error processing data: {exc}"
        return f"<span style='color: red;'>{status}</span>", gr.update(interactive=False), None, None, None, None, None

def on_gnn_model_tab_selected(evt: gr.SelectData):
    return evt.value

def on_create_model(dataset, gnn_model_tab, device, datatype, random_seed,
                    schnet_n_hiddens, schnet_n_filters, schnet_n_interactions, schnet_n_gaussians, schnet_cutoff, schnet_max_num_neighbors, schnet_readout, schnet_dipole,
                    dimenetplusplus_n_hiddens, dimenetplusplus_n_blocks, dimenetplusplus_int_emb_size, dimenetplusplus_basis_emb_size, dimenetplusplus_out_emb_channels, dimenetplusplus_n_spherical, dimenetplusplus_n_radial, dimenetplusplus_cutoff, dimenetplusplus_max_num_neighbors, dimenetplusplus_envelope_exponent, dimenetplusplus_n_output_layers,
                    visnet_n_hiddens, visnet_n_heads, visnet_n_layers, visnet_num_rbf, visnet_trainable_rbf, visnet_max_z, visnet_cutoff, visnet_max_num_neighbors, visnet_vertex, model_name=None):
    
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
        torch.manual_seed(random_seed)
        if device == 'cuda':
            torch.cuda.manual_seed(random_seed)
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
        else:
            raise ValueError(f"Unknown 3D GNN model '{gnn_model_tab}'.")

        # Reset training state
        train_losses = []
        val_losses = []
        trained_epochs = 0
        
        attach_model_run(model, dataset, model_name, random_seed, "regression_3d", run_parameters)
        status = "Model created successfully. Create an optimizer to start training."
        # Training is disabled until an optimizer is created for *this* model: an optimizer
        # captures the parameters it was built with, so a stale one silently updates nothing.
        return f"<span style='color: green;'>{status}</span>", \
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), \
            model, train_losses, val_losses, trained_epochs
    except Exception as exc:
        status = f"Error creating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", \
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), \
            model, train_losses, val_losses, trained_epochs

def on_save_checkpoint(working_directory_path, checkpoint_name, model, trained_epochs, output_scaler):
    try:
        if model is None:
            raise ValueError("Please create a model first.")
        checkpoint_name = sanitize_path_component(checkpoint_name, "checkpoint name")
        # The output scaler denormalises every prediction, so a checkpoint reloaded without it
        # silently returns predictions on the wrong scale.
        checkpoint = {
            'model_class': type(model).__name__,
            'state_dict': model.state_dict(),
            'run_context': getattr(model, '_run_context', None),
            'trained_epochs': trained_epochs,
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

def on_load_checkpoint(working_directory_path, checkpoint_file_name, model, train_losses, val_losses, trained_epochs, output_scaler):
    try:
        if model is None:
            raise ValueError("Please create a model first, so the checkpoint has an architecture to load into.")
        checkpoint_file_path = resolve_saved_model(working_directory_path, checkpoint_file_name, ".ckpt")
        checkpoint, run_note = load_run_checkpoint(model, checkpoint_file_path, checkpoint_file_name, working_directory_path)

        status = f"Checkpoint loaded: {checkpoint_file_name}.{run_note}"
        return f"<span style='color: green;'>{status}</span>", model, [], [], checkpoint.get('trained_epochs', 0), \
            checkpoint.get('output_scaler', output_scaler)
    except Exception as exc:
        status = f"Error loading checkpoint: {exc}"
        # Leave the existing training state untouched - returning None here breaks the next Train click.
        return f"<span style='color: red;'>{status}</span>", model, train_losses, val_losses, trained_epochs, output_scaler

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
def train(model, dataloader, device, datatype, output_scaler, optimizer):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.long)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y.detach().cpu().double().numpy()), device=device, dtype=datatype)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(z, pos, batch_index)
        if isinstance(output, tuple):
            output = output[0]

        # Refuse incompatible targets instead of broadcasting the one-output prediction.
        if output.shape != y_scaled.shape:
            raise ValueError("Model output shape does not match the target shape. Please re-create the model for this dataset.")
        # Compute loss and backpropagate
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
def validation(model, dataloader, device, datatype, output_scaler):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.long)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y_scaled = torch.tensor(output_scaler.transform(batch.y.detach().cpu().double().numpy()), device=device, dtype=datatype)
        
        with torch.no_grad():
            # Forward pass
            output = model(z, pos, batch_index)
            if isinstance(output, tuple):
                output = output[0]

            if output.shape != y_scaled.shape:
                raise ValueError("Model output shape does not match the target shape. Please re-create the model for this dataset.")
            # Compute loss
            loss = loss_fn(output.float(), y_scaled.float()) if datatype in (torch.float16, torch.bfloat16) else loss_fn(output, y_scaled)
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite. Try a lower learning rate or float32 data type.")
            total_loss += loss.item() * y_scaled.shape[0]
        num_samples += y_scaled.shape[0]

    return math.sqrt(total_loss / num_samples) if num_samples > 0 else 0.0

def on_train(model, train_dataloader, val_dataloader, device, datatype, output_scaler, optimizer, scheduler, epochs, trained_epochs, train_losses, val_losses, progress=gr.Progress()):
    loss_epochs = None
    try:
        loss_epochs = list(loss_epoch_numbers(train_losses, val_losses, trained_epochs))
        ensure_optimizer_matches_model(model, optimizer)
        ensure_model_on_device(model, device)
        ensure_model_split(model, train_dataloader)
        if train_dataloader is None or val_dataloader is None:
            raise ValueError("Please process the data first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data first.")

        # Training loop. Both losses are appended together and the epoch counter is bumped
        # only after a full epoch, so a failure part-way through leaves consistent state.
        for _ in progress.tqdm(range(trained_epochs, epochs + trained_epochs), total=epochs, desc="Training"):
            train_loss = train(model, train_dataloader, device, datatype, output_scaler, optimizer)
            val_loss = validation(model, val_dataloader, device, datatype, output_scaler)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            trained_epochs += 1
            loss_epochs.append(trained_epochs)
            scheduler.step()

        status = f"<span style='color: green;'>Training completed for {trained_epochs} epochs.</span>"
    except Exception as exc:
        status = f"<span style='color: red;'>Error during training: {exc}</span>"

    # Plot whatever was completed, so a mid-run failure does not discard finished epochs.
    figure = plt.figure(figsize=(10, 5))
    plt.plot(loss_epochs or [], train_losses if loss_epochs is not None else [], label='Train loss')
    plt.plot(loss_epochs or [], val_losses if loss_epochs is not None else [], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE loss (scaled targets)')
    plt.legend()
    plt.grid(True)
    plt.close(figure)

    export_losses_button = gr.Button(value="Export", interactive=bool(train_losses) and loss_epochs is not None)

    return status, figure, export_losses_button, trained_epochs, train_losses, val_losses

def on_export_losses(working_directory_path, file_name, train_losses, val_losses, trained_epochs=None):
    try:
        if not train_losses:
            raise ValueError("Please train the model first.")
        df = pd.DataFrame({"epoch": loss_epoch_numbers(train_losses, val_losses, trained_epochs), "train_losses": train_losses, "val_losses": val_losses})
        filename = export_result_csv(working_directory_path, file_name, df)
        status = f"Losses exported to {filename}."
        return f"<span style='color: green;'>{html.escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting losses: {exc}"
        return f"<span style='color: red;'>{html.escape(status)}</span>", gr.update()

# Define the test function
@torch.no_grad()
def test(model, dataloader, device, datatype, output_scaler):
    model.eval()
    y_test_list = []
    y_pred_scaled_list = []
    smiles_arr = []
    
    for batch in dataloader:
        # Get input data
        z = batch.z.to(device=device, dtype=torch.long)
        pos = batch.pos.to(device=device, dtype=datatype)
        batch_index = batch.batch.to(device=device)
        
        # Get target values
        y = batch.y
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
    # NumPy does not support bfloat16; float64 also preserves double precision runs.
    y_pred = (output_scaler.inverse_transform(y_pred_scaled.double().numpy())
              if y_pred_scaled.numel() else np.empty((0, 1)))
    return y_test, y_pred, smiles_arr

def on_evaluate(model, dataset, test_dataloader, device, datatype, output_scaler):
    try:
        if model is None or test_dataloader is None:
            raise ValueError("Please process the data and create a model first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data, or load a checkpoint that stores one.")
        ensure_model_on_device(model, device)
        ensure_model_split(model, dataset)
        # Get test predictions
        y_test, y_pred, test_smiles_arr = test(model, test_dataloader, device, datatype, output_scaler)
        if len(test_smiles_arr) == 0:
            raise ValueError("The test split is empty. Please lower the validation size or raise the test size.")
        y_test_np = y_test.detach().cpu().double().numpy().ravel()
        y_pred_np = y_pred.ravel()

        eval_df = pd.DataFrame()
        eval_df['SMILES'] = test_smiles_arr
        target_column = dataset.target_column_name
        eval_df[target_column] = y_test_np.tolist()
        eval_df[f"{target_column} (Prediction)"] = y_pred_np.tolist()

        metrics_df = metrics_from_predictions("regression", eval_df, [target_column]).drop(columns="Target").round({"Value": 4})

        # Visualization
        scatter_plot = plt.figure(figsize=(8, 8))
        plt.scatter(y_test_np, y_pred_np)
        limits = [min(y_test_np.min(), y_pred_np.min()), max(y_test_np.max(), y_pred_np.max())]
        plt.plot(limits, limits, color='navy', lw=1, linestyle='--')
        plt.xlabel('actual value')
        plt.ylabel('predicted value')
        plt.grid(True)
        plt.close(scatter_plot)

        status = f"Evaluation completed."

        return f"<span style='color: green;'>{status}</span>", eval_df, metrics_df, scatter_plot, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as exc:
        status = f"Error evaluating model: {exc}"
        return f"<span style='color: red;'>{status}</span>", None, None, None, gr.update(interactive=False), gr.update(interactive=False)

def on_export_evaluation(working_directory_path, file_name, eval_df):
    try:
        filename = export_result_csv(working_directory_path, file_name, eval_df)
        status = f"Evaluation rows exported: {filename}"
        return f"<span style='color: green;'>{html.escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting evaluation: {exc}"
        return f"<span style='color: red;'>{html.escape(status)}</span>", gr.update()

def on_export_metrics(working_directory_path, file_name, eval_df, dataset):
    try:
        if dataset is None or dataset.target_column_name is None:
            raise ValueError("Please process labeled data and run an evaluation first.")
        metrics = metrics_from_predictions("regression", eval_df, [dataset.target_column_name])
        filename = export_result_csv(working_directory_path, file_name, metrics)
        status = f"Test metrics exported: {filename}"
        return f"<span style='color: green;'>{html.escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting test metrics: {exc}"
        return f"<span style='color: red;'>{html.escape(status)}</span>", gr.update()

def on_process_prediction_data(working_directory_path, prediction_dataset_file, prediction_graph_directory, datatype, batch_size):
    try:
        # Process the prediction dataset and create a dataloader
        prediction_dataset_file_path = os.path.join(working_directory_path, prediction_dataset_file)
        prediction_graph_directory_path = os.path.join(working_directory_path, prediction_graph_directory)                                        
        prediction_dataset = MolecularDataset3D(prediction_dataset_file_path, prediction_graph_directory_path, datatype, target_datatype=torch.float64)
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
        z = batch.z.to(device=device, dtype=torch.long)
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
    # NumPy does not support bfloat16; float64 also preserves double precision runs.
    y_pred = (output_scaler.inverse_transform(y_pred_scaled.double().numpy())
              if y_pred_scaled.numel() else np.empty((0, 1)))
    return smiles_list, y_pred

def on_predict(model, dataset, predict_dataloader, device, datatype, output_scaler):
    try:
        if model is None or predict_dataloader is None:
            raise ValueError("Please process the prediction data first.")
        if output_scaler is None:
            raise ValueError("No output scaler is available. Please process the data, or load a checkpoint that stores one.")
        ensure_model_on_device(model, device)
        # Get predictions
        smiles_list, y_pred = predict(model, predict_dataloader, device, datatype, output_scaler)
        if len(smiles_list) == 0:
            raise ValueError("The prediction dataset is empty.")
        if y_pred.ndim not in (1, 2) or (y_pred.ndim == 2 and y_pred.shape[1] != 1) or y_pred.shape[0] != len(smiles_list):
            raise ValueError("Unexpected prediction shape for a single-target 3D model.")
        y_pred_np = y_pred.ravel()

        prediction_df = pd.DataFrame()
        prediction_df['SMILES'] = smiles_list
        target_column = prediction_target_names(model, dataset, 1)[0]
        prediction_df[f"{target_column} (Prediction)"] = y_pred_np.tolist()

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
        return f"<span style='color: green;'>{html.escape(status)}</span>", get_working_directory_contents(working_directory_path)
    except Exception as exc:
        status = f"Error exporting predictions: {exc}"
        return f"<span style='color: red;'>{html.escape(status)}</span>", gr.update()

def regression_3d_tab_content():
    with gr.Tab("Regression - 3D GNN") as regression_3d_tab:
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
                upload_controls = build_dataset_upload_controls(is_3d=True)
                dataset_file = upload_controls.file
                dataset_name_textbox = upload_controls.name
                smiles_column_textbox = upload_controls.smiles
                target_columns_textbox = upload_controls.targets
                canonicalize_smiles_checkbox = upload_controls.canonicalize_smiles
                remove_duplicates_checkbox = upload_controls.remove_duplicates
                remove_salt_checkbox = upload_controls.remove_salt
                upload_dataset_button = upload_controls.button
                dataset_preview_dataframe = upload_controls.preview
                extraction_controls = feature_extraction.build_feature_extraction_controls(is_3d=True)
                dataset_file_feature_extraction_dropdown = extraction_controls.dataset
                graph_directory_textbox = extraction_controls.graph_directory
                extract_graphs_button = extraction_controls.graph_button
                number_of_conformers_slider = extraction_controls.number_of_conformers
                force_field_dropdown = extraction_controls.force_field
                with gr.Accordion("Load and process data"):
                    with gr.Row():
                        dataset_file_process_data_dropdown = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            graph_directory_dropdown = gr.Dropdown(label="Graph directory", choices=[], info="Extracted graphs in GRAPH_ folders.")
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
                            output_scaler_state = gr.State()
                    split_context_state = gr.State()
                    similarity_controls = build_similarity_controls()
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
                            with gr.Row():
                                model_name_textbox = gr.Textbox(label="Model name", value="regression_3d", info="Saved in MODEL_<model name>_<model seed>.")
                                model_random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Model random seed")
                            create_model_button = gr.Button(value="Create model", interactive=False)
                            run_directory_textbox = gr.Textbox(label="Run directory", interactive=False)
                            model_state = gr.State()
                            train_losses_state = gr.State()
                            val_losses_state = gr.State()
                            trained_epochs_state = gr.State()
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
    concurrency_id = "regression_3d-" + str(split_context_state._id)
    directory_resets = directory_reset_updates(
        clear_states=[split_context_state, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state,
                      model_state, optimizer_state, schedular_state, evaluation_df_state, prediction_dataset_state,
                      prediction_dataloader_state, output_scaler_state],
        loss_states=[train_losses_state, val_losses_state], epoch_state=trained_epochs_state,
        disable_buttons=[create_model_button, create_optimizer_button, train_button, evaluate_button,
                         save_checkpoint_button, load_checkpoint_button, process_prediction_data_button,
                         predict_button, export_losses_button, export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[run_directory_textbox, dataset_preview_dataframe, training_plot, evaluation_metrics_dataframe, evaluation_plot,
                          prediction_dataframe], similarity_controls=similarity_controls,
    )
    directory_refresh_inputs = [working_directory_path_state, dataset_name_textbox, graph_directory_textbox, checkpoint_file_dropdown]
    directory_refresh_outputs = [working_directory_content_list_dataframe,
                                 dataset_file_feature_extraction_dropdown, graph_directory_dropdown, dataset_file_process_data_dropdown, checkpoint_file_dropdown, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown]
    directory_refresh = dict(refresh_callback=on_content_list_change,
                             refresh_inputs=directory_refresh_inputs[1:],
                             refresh_outputs=directory_refresh_outputs, path_output=working_directory_path_state)
    bind_working_directory(working_directory_dropdown, "change", on_open_working_directory, working_directory_dropdown,
                           [working_directory_dropdown, upload_dataset_button, extract_graphs_button, process_data_button, working_directory_path_state, working_directory_content_list_state],
                           reset_updates=directory_resets, concurrency_id=concurrency_id, **directory_refresh)
    bind_working_directory(open_working_directory_button, "click", on_open_working_directory, working_directory_dropdown,
                           [working_directory_dropdown, upload_dataset_button, extract_graphs_button, process_data_button, working_directory_path_state, working_directory_content_list_state],
                           reset_updates=directory_resets, concurrency_id=concurrency_id, **directory_refresh)
    bind_similarity_controls(similarity_controls, split_context_state, status_html, concurrency_id=concurrency_id)
    working_directory_content_list_state.change(on_content_list_change, directory_refresh_inputs,
                                                directory_refresh_outputs, concurrency_id=concurrency_id)
    
    # Settings interaction
    datatype_dropdown.change(on_datatype_change, datatype_dropdown, datatype_state)

    # Load dataset and process data interactions
    bind_artifact_update(upload_dataset_button, on_upload_dataset, [dataset_file, working_directory_path_state, dataset_name_textbox, smiles_column_textbox, target_columns_textbox,
                                                  canonicalize_smiles_checkbox, remove_duplicates_checkbox, remove_salt_checkbox],
                                [status_html, dataset_preview_dataframe, working_directory_content_list_state], dataset_state=dataset_state, reset_updates=directory_resets, kind="upload", concurrency_id=concurrency_id)

    # Feature extraction interactions
    bind_artifact_update(extract_graphs_button, on_extract_graphs, [working_directory_path_state, dataset_file_feature_extraction_dropdown, graph_directory_textbox, number_of_conformers_slider, force_field_dropdown, datatype_state], [status_html, working_directory_content_list_state], dataset_state=dataset_state, reset_updates=directory_resets, kind="graphs", concurrency_id=concurrency_id)
    
    # Process data interactions
    bind_data_processing(
        process_data_button, capture_split_outputs(on_process_data, 2, similarity_controls),
        [working_directory_path_state, dataset_file_process_data_dropdown, graph_directory_dropdown, datatype_state, test_size_slider, val_size_slider, batch_size_dropdown, shuffle_checkbox, data_split_seed_slider, split_method_radio],
        [status_html, create_model_button, dataset_state, train_dataloader_state, val_dataloader_state, test_dataloader_state, output_scaler_state, split_context_state, *similarity_controls.reset_outputs],
        clear_states=[model_state, optimizer_state, schedular_state, evaluation_df_state,
                      prediction_dataset_state, prediction_dataloader_state],
        loss_states=[train_losses_state, val_losses_state], epoch_state=trained_epochs_state, concurrency_id=concurrency_id,
        disable_buttons=[create_optimizer_button, train_button, evaluate_button,
                         save_checkpoint_button, load_checkpoint_button, process_prediction_data_button,
                         predict_button, export_losses_button, export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[run_directory_textbox, training_plot, evaluation_metrics_dataframe, evaluation_plot, prediction_dataframe],
    )
    
    # Create model interactions
    result_reset = dict(
        concurrency_id=concurrency_id,
        clear_states=[evaluation_df_state],
        disable_buttons=[export_evaluation_button, export_metrics_button, export_prediction_button],
        clear_components=[evaluation_metrics_dataframe, evaluation_plot, prediction_dataframe],
    )
    bind_result_invalidation(create_model_button, show_model_run(on_create_model, 7), [dataset_state, gnn_model_tab_state, device_dropdown, datatype_state, model_random_seed_slider,
                                                schnet_n_hiddens, schnet_n_filters, schnet_n_interactions, schnet_n_gaussians, schnet_cutoff, schnet_max_num_neighbors, schnet_readout, schnet_dipole,
                                                dimenetplusplus_n_hiddens, dimenetplusplus_n_blocks, dimenetplusplus_int_emb_size, dimenetplusplus_basis_emb_size, dimenetplusplus_out_emb_channels, dimenetplusplus_n_spherical, dimenetplusplus_n_radial, dimenetplusplus_cutoff, dimenetplusplus_max_num_neighbors, dimenetplusplus_envelope_exponent, dimenetplusplus_n_output_layers,
                                                visnet_n_hiddens, visnet_n_heads, visnet_n_layers, visnet_num_rbf, visnet_trainable_rbf, visnet_max_z, visnet_cutoff, visnet_max_num_neighbors, visnet_vertex, model_name_textbox],
                                                [status_html, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, process_prediction_data_button, train_button, model_state, train_losses_state, val_losses_state, trained_epochs_state, run_directory_textbox], **result_reset)
    bind_checkpoint_save(save_checkpoint_button, on_save_checkpoint, [working_directory_path_state, checkpoint_name_textbox, model_state, trained_epochs_state, output_scaler_state], [status_html, working_directory_content_list_state], checkpoint_file_dropdown, concurrency_id=concurrency_id)
    bind_result_invalidation(load_checkpoint_button, reset_checkpoint_optimizer(show_model_run(on_load_checkpoint, 1, restore_controls=True)), [working_directory_path_state, checkpoint_file_dropdown, model_state, train_losses_state, val_losses_state, trained_epochs_state, output_scaler_state, optimizer_state, schedular_state],
                                 [status_html, model_state, train_losses_state, val_losses_state, trained_epochs_state, output_scaler_state, run_directory_textbox, model_name_textbox, model_random_seed_slider, optimizer_state, schedular_state, train_button, training_plot, export_losses_button], **result_reset)
    
    # Train model interactions
    create_optimizer_button.click(on_create_optimizer, [model_state, optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [status_html, optimizer_state, schedular_state, train_button], concurrency_id=concurrency_id)
    bind_result_invalidation(train_button, on_train, [model_state, train_dataloader_state, val_dataloader_state, device_dropdown, datatype_state, output_scaler_state,
                                  optimizer_state, schedular_state, epochs_slider, trained_epochs_state, train_losses_state, val_losses_state],
                                 [status_html, training_plot, export_losses_button, trained_epochs_state, train_losses_state, val_losses_state], **result_reset)
    export_losses_button.click(route_model_export(on_export_losses), [working_directory_path_state, loss_file_name_textbox, train_losses_state, val_losses_state, trained_epochs_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    
    # Evaluate model interactions
    evaluate_button.click(on_evaluate, [model_state, dataset_state, test_dataloader_state, device_dropdown, datatype_state, output_scaler_state],
                                       [status_html, evaluation_df_state, evaluation_metrics_dataframe, evaluation_plot, export_evaluation_button, export_metrics_button], concurrency_id=concurrency_id)
    export_evaluation_button.click(route_model_export(on_export_evaluation), [working_directory_path_state, evaluation_file_name_textbox, evaluation_df_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    export_metrics_button.click(route_model_export(on_export_metrics), [working_directory_path_state, metrics_file_name_textbox, evaluation_df_state, dataset_state, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
        
    # Make prediction interactions
    process_prediction_data_button.click(on_process_prediction_data, [working_directory_path_state, prediction_dataset_file_dropdown, prediction_graph_directory_dropdown, datatype_state, batch_size_dropdown], [status_html, prediction_dataset_state, prediction_dataloader_state, predict_button], concurrency_id=concurrency_id)
    predict_button.click(on_predict, [model_state, dataset_state, prediction_dataloader_state, device_dropdown, datatype_state, output_scaler_state],
                                     [status_html, prediction_dataframe, export_prediction_button], concurrency_id=concurrency_id)
    export_prediction_button.click(route_model_export(on_export_prediction), [working_directory_path_state, prediction_file_name_textbox, prediction_dataframe, model_state], [status_html, working_directory_content_list_state], concurrency_id=concurrency_id)
    
    return regression_3d_tab
