import os
import math
import pandas as pd
import torch
import gradio as gr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from datasets import MoleculeDatasetForRegressionHybrid, MoleculeDatasetForRegressionPredictionHybrid
from hybrid_models import *

def on_process_data(dataset_file: gr.File, dataset_name_textbox: gr.Textbox, target_column_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, dimensionality_reduction_checkbox: gr.Checkbox, variance_threshold_slider: gr.Slider, pca_num_components_slider: gr.Slider, test_size_slider: gr.Slider, val_size_slider: gr.Slider, batch_size_dropdown: gr.Dropdown, random_seed_slider: gr.Slider):
    try:
        # Process the dataset
        global dataset, train_dataloader, val_dataloader, test_dataloader
        dataset = MoleculeDatasetForRegressionHybrid(
            dataset_file, dataset_name_textbox, target_column_textbox,
            gcn_featurizer_dropdown, mol_featurizer_dropdown,
            dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider
        )

        total_len = len(dataset)
        test_size = int(total_len * test_size_slider)
        val_size = int(total_len * val_size_slider)
        train_size = total_len - test_size - val_size
        if train_size <= 0 or val_size < 0 or test_size < 0:
            gr.Warning("Invalid split sizes. Please adjust test/val sizes.")
            return None, None

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed_slider)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_dropdown)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_dropdown)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_dropdown)

        create_model_button = gr.Button(value="Create model", interactive=True)
        return f"Data processed: {total_len} molecules.", create_model_button
    except Exception as exc:
        gr.Warning(f"Error!\n{exc}")
        return None, None

gnn_model_tab = "GCN"
def on_gnn_model_tab_selected(evt: gr.SelectData):
    if evt.selected: 
        global gnn_model_tab
        gnn_model_tab = evt.value
    
    return

def on_create_model(gcn_n_hiddens: gr.Dropdown, gcn_num_layers: gr.Slider, gcn_dropout: gr.Slider, gcn_n_outputs: gr.Slider,
                    graph_sage_n_hiddens: gr.Dropdown, graph_sage_num_layers: gr.Slider, graph_sage_dropout: gr.Slider, graph_sage_n_outputs: gr.Slider,
                    gin_n_hiddens: gr.Dropdown, gin_num_layers: gr.Slider, gin_dropout: gr.Slider, gin_n_outputs: gr.Slider,
                    gat_n_hiddens: gr.Dropdown, gat_num_layers: gr.Slider, gat_dropout: gr.Slider, gat_n_outputs: gr.Slider,
                    edgecnn_n_hiddens: gr.Dropdown, edgecnn_num_layers: gr.Slider, edgecnn_dropout: gr.Slider, edgecnn_n_outputs: gr.Slider,
                    attentivefp_n_hiddens: gr.Dropdown, attentivefp_num_layers: gr.Slider, attentivefp_num_timesteps: gr.Slider, attentivefp_dropout: gr.Slider, attentivefp_n_outputs: gr.Slider,
                    customgcn_convolutional_layer_name: gr.Dropdown, customgcn_n_hiddens: gr.Dropdown, customgcn_n_layers: gr.Slider, customgcn_n_heads: gr.Slider, customgcn_n_outputs: gr.Slider,
                    mlp_n_hiddens: gr.Dropdown, mlp_n_layers: gr.Slider, mlp_n_outputs: gr.Slider,
                    predictor_n_hiddens: gr.Dropdown, predictor_n_layers: gr.Slider, random_seed_slider: gr.Slider):
    
    gcn_n_inputs = dataset.num_node_features
    edge_dim = dataset.num_edge_features
    mlp_n_inputs = dataset.num_mol_features
    global device, model, train_losses, val_losses, trained_epochs, mol_features_variance_threshold, mol_features_pca, mol_features_scaler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_seed_slider)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed_slider)

    # Model selection logic
    try:
        model = None
        if gnn_model_tab == "GCN":
            model = GCNModel(gcn_n_inputs, gcn_n_hiddens, gcn_num_layers, gcn_n_outputs, gcn_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "GraphSAGE":
            model = GraphSAGEModel(gcn_n_inputs, graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_n_outputs, graph_sage_dropout,
                                   mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                   predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "GIN":
            model = GINModel(gcn_n_inputs, gin_n_hiddens, gin_num_layers, gin_n_outputs, gin_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "GAT":
            model = GATModel(gcn_n_inputs, gat_n_hiddens, gat_num_layers, gat_n_outputs, gat_dropout,
                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                             predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "EdgeCNN":
            model = EdgeCNNModel(gcn_n_inputs, edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_n_outputs, edgecnn_dropout,
                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                 predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "AttentiveFP":
            model = AttentiveFPModel(gcn_n_inputs, attentivefp_n_hiddens, edge_dim, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_n_outputs, attentivefp_dropout,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).float().to(device)
        elif gnn_model_tab == "Custom":
            # Map custom layer names to their constructors and required args
            custom_layers = {
                "GCNConv": lambda: GCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                 predictor_n_hiddens, predictor_n_layers),
                "SAGEConv": lambda: SAGEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                   mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                   predictor_n_hiddens, predictor_n_layers),
                "SGConv": lambda: SGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers),
                "ClusterGCNConv": lambda: ClusterGCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                               predictor_n_hiddens, predictor_n_layers),
                "GraphConv": lambda: GraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                     predictor_n_hiddens, predictor_n_layers),
                "LEConv": lambda: LEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers),
                "EGConv": lambda: EGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers),
                "MFConv": lambda: MFConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers),
                "TAGConv": lambda: TAGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                 predictor_n_hiddens, predictor_n_layers),
                "ARMAConv": lambda: ARMAConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                   mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                   predictor_n_hiddens, predictor_n_layers),
                "FiLMConv": lambda: FiLMConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                                   mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                   predictor_n_hiddens, predictor_n_layers),
                "PDNConv": lambda: PDNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim, gcn_n_hiddens,
                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                 predictor_n_hiddens, predictor_n_layers),
                "GENConv": lambda: GENConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                 predictor_n_hiddens, predictor_n_layers),
                "ResGatedGraphConv": lambda: ResGatedGraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                                     predictor_n_hiddens, predictor_n_layers),
                "GATConv": lambda: GATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                 predictor_n_hiddens, predictor_n_layers),
                "GATv2Conv": lambda: GATv2ConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                     predictor_n_hiddens, predictor_n_layers),
                "SuperGATConv": lambda: SuperGATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                                           mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                           predictor_n_hiddens, predictor_n_layers),
                "TransformerConv": lambda: TransformerConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                                                 mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                                 predictor_n_hiddens, predictor_n_layers),
                "GeneralConv": lambda: GeneralConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                                         mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                         predictor_n_hiddens, predictor_n_layers),
            }
            if customgcn_convolutional_layer_name in custom_layers:
                model = custom_layers[customgcn_convolutional_layer_name]().to(device)
            else:
                raise ValueError(f"Unknown custom GCN layer: {customgcn_convolutional_layer_name}")

        if model is None:
            raise ValueError("Model could not be instantiated.")

    except Exception as exc:
        gr.Warning(str(exc))
        model = None
        save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
        load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
        create_optimizer_button = gr.Button(value="Create optimizer", interactive=False)
        evaluate_button = gr.Button(value="Evaluate", interactive=False)
        predict_button = gr.Button(value="Predict", interactive=False)
        return None, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button

    # Reset training state
    train_losses = []
    val_losses = []
    trained_epochs = 0

    # Set up feature transformers
    if dataset.dimensionality_reduction:
        mol_features_variance_threshold = dataset.variance_threshold
        mol_features_pca = dataset.pca
    mol_features_scaler = dataset.mol_features_scaler

    # Prepare a sample for model summary
    graph_data = dataset[0]
    x = graph_data.x.float().to(device)
    edge_index = graph_data.edge_index.long().to(device)
    batch_index = torch.tensor([0]).to(device)
    mol_features = graph_data.mol_features.numpy()
    if dataset.dimensionality_reduction:
        mol_features = mol_features_variance_threshold.transform(mol_features)
        mol_features = mol_features_pca.transform(mol_features)
    mol_features_scaled = mol_features_scaler.transform(mol_features)
    mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)

    # Create UI buttons
    save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=True)
    load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=True)
    create_optimizer_button = gr.Button(value="Create optimizer", interactive=True)
    evaluate_button = gr.Button(value="Evaluate", interactive=True)
    predict_button = gr.Button(value="Predict", interactive=True)

    # Model summary
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in ["PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    if need_edge_attr:
        edge_attr = graph_data.edge_attr.float().to(device)
        model_summary = summary(model, x, edge_index, edge_attr, batch_index, mol_features_scaled)
    else:
        model_summary = summary(model, x, edge_index, batch_index, mol_features_scaled)

    return model_summary, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button

trained_epochs = 0
def on_save_checkpoint(model_name_textbox: gr.Textbox):
    checkpoint = {
        'state_dict': model.state_dict()
    }
    checkpoint_dir = './Checkpoints'
    checkpoint_name = f'{model_name_textbox}_{trained_epochs}.ckpt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir + '/' + checkpoint_name)
    return f'Model was saved to {checkpoint_dir}/{checkpoint_name}.'

def on_load_checkpoint(checkpoint_path: gr.File):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    global train_losses, val_losses, trained_epochs
    train_losses = []
    val_losses = []
    trained_epochs = 0

    return f'Checkpoint loaded: {os.path.basename(checkpoint_path)}.'

loss_fn = torch.nn.MSELoss()
def on_create_optimizer(optimizer_dropdown: gr.Dropdown, learning_rate_slider: gr.Slider, learning_rate_decay_slider: gr.Slider):
    global optimizer, scheduler, train_losses, val_losses, trained_epochs
    optimizers = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD
    }
    opt_class = optimizers.get(optimizer_dropdown)
    if not opt_class:
        gr.Warning(f"Unknown optimizer: {optimizer_dropdown}")
        return "Unknown optimizer.", gr.Button(value="Train", interactive=False)

    optimizer = opt_class(model.parameters(), lr=learning_rate_slider)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay_slider)

    train_losses = []
    val_losses = []
    trained_epochs = 0

    train_button = gr.Button(value="Train", interactive=True)
    return "Optimizer created.", train_button

# Define the train function
def train(dataloader, customgcn_convolutional_layer_name):
    model.train()
    total_loss = 0.0
    num_batches = 0
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        if dataset.dimensionality_reduction:
            mol_features = mol_features_variance_threshold.transform(mol_features)
            mol_features = mol_features_pca.transform(mol_features)
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)

        optimizer.zero_grad()
        if need_edge_attr:
            edge_attr = batch.edge_attr.float().to(device)
            output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
        else:
            output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += torch.sqrt(loss).item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

# Define the validation function
@torch.no_grad()
def validation(dataloader, customgcn_convolutional_layer_name):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        if dataset.dimensionality_reduction:
            mol_features = mol_features_variance_threshold.transform(mol_features)
            mol_features = mol_features_pca.transform(mol_features)
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)

        if need_edge_attr:
            edge_attr = batch.edge_attr.float().to(device)
            output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
        else:
            output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
        loss = loss_fn(output, y)
        total_loss += torch.sqrt(loss).item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

def on_train(epochs_slider: gr.Slider, customgcn_convolutional_layer_name: gr.Dropdown, progress=gr.Progress()):
    global trained_epochs
    epochs = epochs_slider
    t = progress.tqdm(range(trained_epochs + 1, epochs + trained_epochs + 1), total=epochs, desc="Training")
    for _ in t:
        train_loss = train(train_dataloader, customgcn_convolutional_layer_name)
        train_losses.append(train_loss)
        val_loss = validation(val_dataloader, customgcn_convolutional_layer_name)
        val_losses.append(val_loss)
        scheduler.step()
        trained_epochs += 1

    # Plot the training and validation loss
    figure = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.grid(True)
    plt.close()

    export_losses_button = gr.Button(value="Export", interactive=True)
    return figure, export_losses_button

def on_export_losses():
    df = pd.DataFrame()
    df['train_losses'] = train_losses
    df['val_losses'] = val_losses
    file_path = f'./{dataset.dataset_name}_{trained_epochs}_losses.csv'
    df.to_csv(file_path)
    return f'Losses exported to {file_path}.'

# Define the test function
@torch.no_grad()
def test(dataloader, customgcn_convolutional_layer_name):
    model.eval()
    y_test_list = []
    y_pred_list = []
    smiles_arr = []
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        if dataset.dimensionality_reduction:
            mol_features = mol_features_variance_threshold.transform(mol_features)
            mol_features = mol_features_pca.transform(mol_features)
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)
        smiles_arr.extend(batch.smiles)

        if need_edge_attr:
            edge_attr = batch.edge_attr.float().to(device)
            output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
        else:
            output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)

        y_test_list.append(y.cpu())
        y_pred_list.append(output.cpu())

    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.tensor([])
    y_pred = torch.cat(y_pred_list, dim=0) if y_pred_list else torch.tensor([])
    return y_test, y_pred, smiles_arr

def on_evaluate(convolutional_layer_name: gr.Dropdown):
    output_scaler = dataset.output_scaler
    global test_smiles_arr, y_test, y_pred
    y_test_scaled, y_pred_scaled, test_smiles_arr = test(test_dataloader, convolutional_layer_name)

    # Handle empty test set
    if y_test_scaled.numel() == 0 or y_pred_scaled.numel() == 0:
        gr.Warning("No test data available for evaluation.")
        return ["No test data available.", None, gr.Button(value="Export", interactive=False)]

    y_test = output_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    # Evaluate the model using different metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    html = f'''
    <div>
        <table>
            <thead>
                <tr><td><b>Metric</b></td><td><b>Value</b></td></tr>
            </thead>
            <tbody>
                <tr><td>Mean absolute error</td><td>{mae:.4f}</td></tr>
                <tr><td>Mean squared error</td><td>{mse:.4f}</td></tr>
                <tr><td>Root mean squared error</td><td>{rmse:.4f}</td></tr>
                <tr><td>Coefficient of determination</td><td>{r2:.4f}</td></tr>
            </tbody>
        </table>
    </div>
    '''

    # Visualization
    scatter_plot = plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred)
    plt.xlabel('actual value')
    plt.ylabel('predicted value')
    plt.grid(True)
    plt.close()

    export_scatter_plot_button = gr.Button(value="Export", interactive=True)
    return [html, scatter_plot, export_scatter_plot_button]

def on_export_scatter_plot():
    df = pd.DataFrame()
    df['y_test'] = y_test.tolist()
    df['y_pred'] = y_pred.tolist()
    df['SMILES'] = test_smiles_arr
    file_path = f'./{dataset.dataset_name}_{trained_epochs}_scatter_plot.csv'
    df.to_csv(file_path)

    return f'Scatter plot exported to {file_path}.'

def on_process_prediction_data(prediction_dataset_file: gr.File, prediction_dataset_name_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, batch_size_dropdown: gr.Dropdown):
    try:
        global prediction_dataset, prediction_dataloader
        prediction_dataset = MoleculeDatasetForRegressionPredictionHybrid(
            prediction_dataset_file, prediction_dataset_name_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown
        )
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size_dropdown)
        return f'Data processed: {len(prediction_dataset)} molecules.'
    except Exception as exc:
        gr.Warning(f"Error processing prediction data!\n{exc}")
        return "Error processing prediction data."

# Define the predict function
@torch.no_grad()
def predict(dataloader, customgcn_convolutional_layer_name):
    model.eval()
    smiles_list = []
    y_pred_list = []
    need_edge_attr = (gnn_model_tab in ["GAT", "AttentiveFP"]) or (
        gnn_model_tab == "Custom" and customgcn_convolutional_layer_name in [
            "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GeneralConv"])
    for batch in dataloader:
        smiles_list.extend(batch.smiles)
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        if dataset.dimensionality_reduction:
            mol_features = mol_features_variance_threshold.transform(mol_features)
            mol_features = mol_features_pca.transform(mol_features)
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)

        if need_edge_attr:
            edge_attr = batch.edge_attr.float().to(device)
            output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
        else:
            output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)

        y_pred_list.append(output.cpu())
    y_pred = torch.cat(y_pred_list, dim=0) if y_pred_list else torch.tensor([])
    return smiles_list, y_pred

def on_predict(customgcn_convolutional_layer_name: gr.Dropdown):
    output_scaler = dataset.output_scaler
    global prediction_df
    smiles_list, y_pred_scaled = predict(prediction_dataloader, customgcn_convolutional_layer_name)

    if y_pred_scaled.numel() == 0:
        gr.Warning("No prediction data available.")
        return pd.DataFrame(), gr.Button(value="Export", interactive=False)

    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    prediction_df = pd.DataFrame({
        'SMILES': smiles_list,
        dataset.target_column: y_pred.tolist()
    })
    export_prediction_button = gr.Button(value="Export", interactive=True)
    return prediction_df, export_prediction_button

def on_export_prediction():
    file_path = f'./{prediction_dataset.dataset_name}_prediction.csv'
    prediction_df.to_csv(file_path)
    return f'Prediction exported to {file_path}.'

def hybrid_gnn_regression_tab_content():
    with gr.Tab("Hybrid GNN Models for Regression") as gnn_regression_tab:
        with gr.Accordion("Dataset"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Dataset file")
                    dataset_name_textbox = gr.Textbox(label="Dataset name", placeholder="dataset", value="dataset")
                    target_column_textbox = gr.Textbox(label="Target column")
                with gr.Column(scale=1):
                    gcn_featurizer_dropdown = gr.Dropdown(label="Graph featurizer", value="MolGraphConvFeaturizer", choices=["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"])
                    mol_featurizer_dropdown = gr.Dropdown(label="Molecule featurizer", value="Mordred descriptors", choices=["Mordred descriptors", "RDKit descriptors", "MACCS keys", "Morgan fingerprint", "Avalon fingerprint", "Atom-pairs fingerprint", "Topological-torsion fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint"])
                    dimensionality_reduction_checkbox = gr.Checkbox(label="Dimensionality reduction", value=False)
                    variance_threshold_slider = gr.Slider(minimum=0, maximum=1.0, value=0.01, step=0.01, label="Variance threshold")
                    pca_num_components_slider = gr.Slider(minimum=1, maximum=256, value=32, step=1, label="PCA number of components")
                with gr.Column(scale=1):
                    test_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Test size")
                    val_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Validation size")
                    batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
                    random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Random seed")
                    process_data_button = gr.Button(value="Process data")
                    process_data_markdown = gr.Markdown()
        with gr.Accordion("GNN model"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Graph convolutional block")
                        with gr.Tabs():
                            with gr.Tab("GCN") as gcn_tab:
                                gcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                gcn_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                gcn_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                gcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                gcn_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("GraphSAGE") as graph_sage_tab:
                                graph_sage_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                graph_sage_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                graph_sage_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                graph_sage_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                graph_sage_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("GIN") as gin_tab:
                                gin_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                gin_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                gin_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                gin_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                gin_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("GAT") as gat_tab:
                                gat_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                gat_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                gat_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                gat_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                gat_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("EdgeCNN") as edgecnn_tab:
                                edgecnn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                edgecnn_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                edgecnn_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                edgecnn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                edgecnn_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("AttentiveFP") as attentivefp_tab:
                                attentivefp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                attentivefp_num_layers = gr.Slider(label="num_layers", value=3, minimum=1, maximum=10, step=1)
                                attentivefp_num_timesteps = gr.Slider(label="num_timesteps", value=3, minimum=1, maximum=10, step=1)
                                attentivefp_dropout = gr.Slider(label="dropout", value=0, minimum=0, maximum=1, step=0.01)
                                attentivefp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                attentivefp_tab.select(on_gnn_model_tab_selected, [], [])
                            with gr.Tab("Custom") as customgcn_tab:
                                customgcn_convolutional_layer_name = gr.Dropdown(label="Graph convolutional layer", value="GCNConv", choices=["GCNConv", "SAGEConv", "SGConv", "ClusterGCNConv", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "SuperGATConv", "TransformerConv", "GeneralConv"])
                                customgcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                                customgcn_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                                customgcn_n_heads = gr.Slider(label="n_heads", minimum=1, maximum=8, value=3, step=1)
                                customgcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                                customgcn_tab.select(on_gnn_model_tab_selected, [], [])
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Molecular feature block")
                        mlp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                        mlp_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                        mlp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                    with gr.Group():
                        gr.Markdown(value="Predictor block")
                        predictor_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                        predictor_n_layers = gr.Slider(label="n_layers", minimum=0, maximum=6, value=3, step=1)
                with gr.Column(scale=1):
                    create_model_button = gr.Button(value="Create model", interactive=False)
                    model_name_textbox = gr.Textbox(label="Model name", placeholder="model", value="model")
                    save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
                    save_checkpoint_markdown = gr.Markdown()
                    checkpoint_file = gr.File(file_types=['.ckpt'], type='filepath', label="Checkpoint file")
                    load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
                    load_checkpoint_markdown = gr.Markdown()
            with gr.Row(equal_height=True):
                model_summary_textarea = gr.TextArea(label="Model summary", elem_classes="monospaced")
        with gr.Accordion("Model training"):
            with gr.Row(equal_height=True): 
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Optimizer")
                        optimizer_dropdown = gr.Dropdown(label="Optimizer", value="Adam", choices=["Adam", "AdamW", "SGD"]) 
                        learning_rate_slider = gr.Slider(label="Learning rate", minimum=0.00001, maximum=0.1, value=0.001, step=0.00001) 
                        learning_rate_decay_slider = gr.Slider(label="Learning rate decay", minimum=0.5, maximum=1, value=0.99, step=0.001)
                    create_optimizer_button = gr.Button(value="Create optimizer", interactive=False)
                    create_optimizer_markdown = gr.Markdown()
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
                            export_losses_button = gr.Button(value="Export", interactive=False)
                            export_losses_markdown = gr.Markdown()
        with gr.Accordion("Model evaluation"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    evaluate_button = gr.Button(value="Evaluate", interactive=False)
                    evaluation_metrics_html = gr.HTML()     
                    export_scatter_plot_button = gr.Button(value="Export", interactive=False)
                    export_scatter_plot_markdown = gr.Markdown()
                with gr.Column(scale=2):
                    evaluation_plot = gr.Plot()
        with gr.Accordion("Make prediction"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    prediction_dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Prediction dataset file")
                    prediction_dataset_name_textbox = gr.Textbox(label="Dataset name", placeholder="prediction_dataset", value="prediction_dataset")
                    process_prediction_data_button = gr.Button(value="Process prediction data")
                    process_prediction_data_markdown = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Row(equal_height=True): 
                        with gr.Column(scale=1):
                            predict_button = gr.Button(value="Predict", interactive=False)
                        with gr.Column(scale=1):
                            export_prediction_button = gr.Button(value="Export", interactive=False)
                            export_prediction_markdown = gr.Markdown()
                    with gr.Row(): 
                        prediction_datatable = gr.DataFrame(label="Prediction", wrap=False, interactive=False)

    process_data_button.click(on_process_data, [dataset_file, dataset_name_textbox, target_column_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown, dimensionality_reduction_checkbox, variance_threshold_slider, pca_num_components_slider, test_size_slider, val_size_slider, batch_size_dropdown, random_seed_slider],
                              [process_data_markdown, create_model_button])
    create_model_button.click(on_create_model, [gcn_n_hiddens, gcn_num_layers, gcn_dropout, gcn_n_outputs,
                                                graph_sage_n_hiddens, graph_sage_num_layers, graph_sage_dropout, graph_sage_n_outputs,
                                                gin_n_hiddens, gin_num_layers, gin_dropout, gin_n_outputs,
                                                gat_n_hiddens, gat_num_layers, gat_dropout, gat_n_outputs,
                                                edgecnn_n_hiddens, edgecnn_num_layers, edgecnn_dropout, edgecnn_n_outputs,
                                                attentivefp_n_hiddens, attentivefp_num_layers, attentivefp_num_timesteps, attentivefp_dropout, attentivefp_n_outputs,
                                                customgcn_convolutional_layer_name, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                                mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                predictor_n_hiddens, predictor_n_layers, random_seed_slider],
                                                [model_summary_textarea, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button])
    save_checkpoint_button.click(on_save_checkpoint, model_name_textbox, save_checkpoint_markdown)
    load_checkpoint_button.click(on_load_checkpoint, checkpoint_file, load_checkpoint_markdown)
    create_optimizer_button.click(on_create_optimizer, [optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [create_optimizer_markdown, train_button])
    train_button.click(on_train, [epochs_slider, customgcn_convolutional_layer_name], [training_plot, export_losses_button])
    export_losses_button.click(on_export_losses, [], export_losses_markdown)
    evaluate_button.click(on_evaluate, customgcn_convolutional_layer_name, [evaluation_metrics_html, evaluation_plot, export_scatter_plot_button])
    export_scatter_plot_button.click(on_export_scatter_plot, [], export_scatter_plot_markdown)
    process_prediction_data_button.click(on_process_prediction_data, [prediction_dataset_file, prediction_dataset_name_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown, batch_size_dropdown], process_prediction_data_markdown)
    predict_button.click(on_predict, customgcn_convolutional_layer_name, [prediction_datatable, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [], export_prediction_markdown)
    
    return gnn_regression_tab