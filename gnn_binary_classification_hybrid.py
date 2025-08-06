import os
import pandas as pd
import torch
import gradio as gr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from matplotlib import pyplot as plt
from datasets import MoleculeDatasetForBinaryClassificationHybrid, MoleculeDatasetForBinaryClassificationPredictionHybrid
from hybrid_models import *

def on_process_data(dataset_file: gr.File, dataset_name_textbox: gr.Textbox, target_column_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, dimensionality_reduction_checkbox: gr.Checkbox, variance_threshold_slider: gr.Slider, pca_num_components_slider: gr.Slider, test_size_slider: gr.Slider, val_size_slider: gr.Slider, batch_size_dropdown: gr.Dropdown, random_seed_slider: gr.Slider):
    try:
        global dataset, train_dataloader, val_dataloader, test_dataloader
        dataset = MoleculeDatasetForBinaryClassificationHybrid(
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
        else: # Custom
            if customgcn_convolutional_layer_name == "GCNConv":
                model = GCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "SAGEConv":
                model = SAGEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "SGConv":
                model = SGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "ClusterGCNConv":
                model = ClusterGCNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                            predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "GraphConv":
                model = GraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                       mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                       predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "LEConv":
                model = LEConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "EGConv":
                model = EGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "MFConv":
                model = MFConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                    mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                    predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "TAGConv":
                model = TAGConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "ARMAConv":
                model = ARMAConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "FiLMConv":
                model = FiLMConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs,
                                      mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                      predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "PDNConv":
                model = PDNConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim, gcn_n_hiddens,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "GENConv":
                model = GENConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "ResGatedGraphConv":
                model = ResGatedGraphConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_outputs, edge_dim,
                                               mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                               predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "GATConv":
                model = GATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                     mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                     predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "GATv2Conv":
                model = GATv2ConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                       mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                       predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "SuperGATConv":
                model = SuperGATConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs,
                                          mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                          predictor_n_hiddens, predictor_n_layers).to(device)
            elif customgcn_convolutional_layer_name == "TransformerConv":
                model = TransformerConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                             mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                             predictor_n_hiddens, predictor_n_layers).to(device)
            else: # customgcn_convolutional_layer_name == "GeneralConv":
                model = GeneralConvModel(gcn_n_inputs, customgcn_n_hiddens, customgcn_n_layers, customgcn_n_heads, customgcn_n_outputs, edge_dim,
                                         mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                         predictor_n_hiddens, predictor_n_layers).to(device)

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
    except Exception as exc:
        gr.Warning(str(exc))
        model = None
        save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
        load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
        create_optimizer_button = gr.Button(value="Create optimizer", interactive=False)
        evaluate_button = gr.Button(value="Evaluate", interactive=False)
        predict_button = gr.Button(value="Predict", interactive=False)
        return None, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button

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

loss_fn = torch.nn.BCEWithLogitsLoss()
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
        return None, gr.Button(value="Train", interactive=False)

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

        loss = loss_fn(output, y.reshape(-1, 1))
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

        with torch.no_grad():
            if need_edge_attr:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            else:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)

            loss = loss_fn(output, y.reshape(-1, 1))
            total_loss += torch.sqrt(loss).item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

def on_train(epochs_slider: gr.Slider, gcn_model_name: gr.Dropdown, progress=gr.Progress()):
    global trained_epochs
    epochs = epochs_slider
    t = progress.tqdm(range(trained_epochs + 1, epochs + trained_epochs + 1), total=epochs, desc="Training")
    for _ in t:
        train_loss = train(train_dataloader, gcn_model_name)
        train_losses.append(train_loss)
        val_loss = validation(val_dataloader, gcn_model_name)
        val_losses.append(val_loss)
        scheduler.step()
        trained_epochs += 1

    # Plot the training and validation loss
    figure = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE with logits loss')
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
    logits_list = []
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
        smiles = batch.smiles

        with torch.no_grad():
            if need_edge_attr:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            else:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)

            y_test_list.append(y.cpu().detach())
            logits_list.append(output.cpu().detach())
            smiles_arr.extend(smiles)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.tensor([])
    logits = torch.cat(logits_list, dim=0) if logits_list else torch.tensor([])
    return y_test, logits, smiles_arr

def on_evaluate(gcn_model_name: gr.Dropdown):
    global y_test, y_pred, test_smiles_arr
    y_test, logits, test_smiles_arr = test(test_dataloader, gcn_model_name)
    probabilities = torch.sigmoid(logits)
    y_pred = torch.round(probabilities).int().detach().numpy().reshape(-1)
    y_test_np = y_test.int().detach().numpy()

    # Handle empty test set
    if y_test_np.size == 0 or y_pred.size == 0:
        gr.Warning("No test data available for evaluation.")
        return ["No test data available.", None, None, gr.Button(value="Export", interactive=False)]

    # Evaluate the model using different metrics
    cm = confusion_matrix(y_test_np, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_figure, ax = plt.subplots(figsize=(6, 6))
    display.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title('Confusion Matrix')

    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred)
    recall = recall_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    TN = cm[0, 0] if cm.shape[0] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    roc_auc = roc_auc_score(y_test_np, y_pred)

    html = f'''
    <div>
        <table>
            <thead>
                <tr><td><b>Metric</b></td><td><b>Value</b></td></tr>
            </thead>
            <tbody>
                <tr><td>Accuracy</td><td>{accuracy:.4f}</td></tr>
                <tr><td>Precision</td><td>{precision:.4f}</td></tr>
                <tr><td>Recall</td><td>{recall:.4f}</td></tr>
                <tr><td>Specificity</td><td>{specificity:.4f}</td></tr>
                <tr><td>F1 score</td><td>{f1:.4f}</td></tr>
                <tr><td>ROC-AUC score</td><td>{roc_auc:.4f}</td></tr>
            </tbody>
        </table>
    </div>
    '''

    # Generate ROC curve values
    fpr, tpr, _ = roc_curve(y_test_np, y_pred)
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

    export_evaluation_button = gr.Button(value="Export", interactive=True)
    return [html, cm_figure, roc_auc_figure, export_evaluation_button]

def on_export_evaluation():
    df = pd.DataFrame()
    df['y_test'] = y_test.tolist()
    df['y_pred'] = y_pred.tolist()
    df['SMILES'] = test_smiles_arr
    file_path = f'./{dataset.dataset_name}_{trained_epochs}_eval.csv'
    df.to_csv(file_path)
    return f'Evaluation exported to {file_path}.'

def on_process_prediction_data(prediction_dataset_file: gr.File, prediction_dataset_name_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, batch_size_dropdown: gr.Dropdown):
    try:
        global prediction_dataset, prediction_dataloader
        prediction_dataset = MoleculeDatasetForBinaryClassificationPredictionHybrid(
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
    logits_list = []
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

        with torch.no_grad():
            if need_edge_attr:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            else:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
            logits_list.append(output.cpu().detach())
    logits = torch.cat(logits_list, dim=0) if logits_list else torch.tensor([])
    return smiles_list, logits

def on_predict(gcn_model_name: gr.Dropdown):
    smiles_list, logits = predict(prediction_dataloader, gcn_model_name)
    probabilities = torch.sigmoid(logits)
    y_pred = torch.round(probabilities).int().detach().numpy().reshape(-1)

    global prediction_df
    if y_pred.size == 0 or len(smiles_list) == 0:
        gr.Warning("No prediction data available.")
        return None, gr.Button(value="Export", interactive=False)

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

def hybrid_gnn_binary_classification_tab_content():
    with gr.Tab("Hybrid GNN Models for Binary Classification") as gnn_binary_classification_tab:
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
                                customgcn_convolutional_layer_name = gr.Dropdown(label="Graph convolutional layer", value="GCNConv", choices=["GCNConv", "SAGEConv", "CuGraphSAGEConv", "SGConv", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "SuperGATConv", "TransformerConv", "GeneralConv"])
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
                    export_evaluation_button = gr.Button(value="Export", interactive=False)
                    export_evaluation_markdown = gr.Markdown()
                with gr.Column(scale=1):
                    confusion_matrix_plot = gr.Plot()
                with gr.Column(scale=1):
                    roc_auc_curve_plot = gr.Plot()
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
    evaluate_button.click(on_evaluate, customgcn_convolutional_layer_name, [evaluation_metrics_html, confusion_matrix_plot, roc_auc_curve_plot, export_evaluation_button])
    export_evaluation_button.click(on_export_evaluation, [], export_evaluation_markdown)
    process_prediction_data_button.click(on_process_prediction_data, [prediction_dataset_file, prediction_dataset_name_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown, batch_size_dropdown], process_prediction_data_markdown)
    predict_button.click(on_predict, customgcn_convolutional_layer_name, [prediction_datatable, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [], export_prediction_markdown)
    
    return gnn_binary_classification_tab