import os
import math
import numpy as np
import pandas as pd
import torch
import gradio as gr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from datasets import *
from models import *

def on_process_data(dataset_file: gr.File, dataset_name_textbox: gr.Textbox, target_column_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, test_size_slider: gr.Slider, val_size_slider: gr.Slider, batch_size_dropdown: gr.Dropdown, random_seed_slider: gr.Slider):
    # Load dataset
    dataset_file_path = dataset_file
    dataset_name = dataset_name_textbox
    target_column = target_column_textbox
    gcn_featurizer_name = gcn_featurizer_dropdown
    mol_featurizer_name = mol_featurizer_dropdown

    try:
        # Process the dataset
        global dataset
        dataset = MoleculeDatasetForRegression(dataset_file_path, dataset_name, target_column, gcn_featurizer_name, mol_featurizer_name)
        
        # Train-validation-test splitting
        dataset = dataset.shuffle()

        test_size = int(len(dataset) * test_size_slider)
        val_size = int(len(dataset) * val_size_slider)
        train_size = len(dataset) - test_size - val_size

        torch.manual_seed(random_seed_slider)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed_slider)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # DataLoaders
        global train_dataloader
        global test_dataloader
        global val_dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_dropdown)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_dropdown)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_dropdown)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc.args))
        return None, None
    
    create_model_button = gr.Button(value="Create model", interactive=True)
    return f"Data processed: {len(dataset)} molecules.", create_model_button

def on_create_model(gcn_model_name: gr.Dropdown, gcn_n_hiddens: gr.Dropdown, gcn_n_layers: gr.Slider, gcn_n_heads: gr.Slider, gcn_n_outputs: gr.Slider,
                    mlp_n_hiddens: gr.Dropdown, mlp_n_layers: gr.Slider, mlp_n_outputs: gr.Slider,
                    predictor_n_hiddens: gr.Dropdown, predictor_n_layers: gr.Slider, random_seed_slider: gr.Slider):
    
    gcn_n_inputs = dataset.num_node_features
    edge_dim = dataset.num_edge_features
    mlp_n_inputs = dataset.num_mol_features
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_seed_slider)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed_slider)

    global model
    try:
        if gcn_model_name == "GCN":
            model = GCNModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GraphSAGE":
            model = GraphSAGEModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "SGConv":
            model = SGConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "ClusterGCN":
            model = ClusterGCNModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GraphConv":
            model = GraphConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "LEConv":
            model = LEConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "EGConv":
            model = EGConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "MFConv":
            model = MFConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "TAGConv":
            model = TAGConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "ARMAConv":
            model = ARMAConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "FiLMConv":
            model = FiLMConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "PDNConv":
            model = PDNConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs, edge_dim, gcn_n_hiddens,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GENConv":
            model = GENConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs, edge_dim,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "ResGatedGraphConv":
            model = ResGatedGraphConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_outputs, edge_dim,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GAT":
            model = GATModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GATv2":
            model = GATv2Model(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs, edge_dim,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "SuperGAT":
            model = SuperGATModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "TransformerConv":
            model = TransformerConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs, edge_dim,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
        elif gcn_model_name == "GeneralConv":
            model = GeneralConvModel(gcn_n_inputs, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs, edge_dim,
                            mlp_n_inputs, mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                            predictor_n_hiddens, predictor_n_layers).to(device)
    except Exception as exc:
        gr.Warning(str(exc))
        model = None
        save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
        load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
        create_optimizer_button = gr.Button(value="Create optimizer", interactive=False)
        evaluate_button = gr.Button(value="Evaluate", interactive=False)
        predict_button = gr.Button(value="Predict", interactive=False)
        return None, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button
    
    global train_losses
    global val_losses
    global trained_epochs
    train_losses = []
    val_losses = []
    trained_epochs = 0

    global mol_features_scaler
    mol_features_arr = np.loadtxt('.\\Datasets\\' + dataset.dataset_name + '\\raw\\mol_features.csv', delimiter=",")
    mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
    mol_features_scaler.fit(mol_features_arr)

    graph_data = dataset[0]
    x = graph_data.x.float().to(device)
    edge_index = graph_data.edge_index.long().to(device)
    batch_index = torch.tensor([0]).to(device)
    mol_features = graph_data.mol_features.numpy()
    mol_features_scaled = mol_features_scaler.transform(mol_features)
    mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
    
    save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=True)
    load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=True)
    create_optimizer_button = gr.Button(value="Create optimizer", interactive=True)
    evaluate_button = gr.Button(value="Evaluate", interactive=True)
    predict_button = gr.Button(value="Predict", interactive=True)

    if gcn_model_name in ["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "SuperGATConv"]:
        return summary(model, x, edge_index, batch_index, mol_features_scaled), save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button
    else:
        edge_attr = graph_data.edge_attr.float().to(device)
        return summary(model, x, edge_index, edge_attr, batch_index, mol_features_scaled), save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button

trained_epochs = 0
def on_save_checkpoint(model_name_textbox: gr.Textbox):
    checkpoint = {
        'state_dict': model.state_dict()
    }
    checkpoint_dir = '.\\Checkpoints'
    checkpoint_name = f'{model_name_textbox}_{trained_epochs}.ckpt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir + '\\' + checkpoint_name)
    return f'Model was saved to {checkpoint_dir}\\{checkpoint_name}.'

def on_load_checkpoint(checkpoint_path: gr.File):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    global train_losses
    global val_losses
    global trained_epochs
    train_losses = []
    val_losses = []
    trained_epochs = 0

    return f'Checkpoint loaded: {os.path.basename(checkpoint_path)}.'

loss_fn = torch.nn.MSELoss()
def on_create_optimizer(optimizer_dropdown: gr.Dropdown, learning_rate_slider: gr.Slider, learning_rate_decay_slider: gr.Slider):
    global optimizer
    if optimizer_dropdown == "Adam":
        optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_slider)
    elif optimizer_dropdown == "AdamW":
        optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_slider)
    elif optimizer_dropdown == "SGD":
        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_slider)
        
    global scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay_slider)

    global train_losses
    global val_losses
    global trained_epochs
    train_losses = []
    val_losses = []
    trained_epochs = 0

    train_button = gr.Button(value="Train", interactive=True)

    return "Optimizer created.", train_button

# Define the train function
def train(dataloader, gcn_model_name):
    total_loss = 0
    batch_index = 0
    model.train()
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)
        if gcn_model_name in ["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "SuperGATConv"]:
            output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
        else:
            edge_attr = batch.edge_attr.float().to(device)
            output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += torch.sqrt(loss).cpu().detach().item()
        batch_index += 1
    return total_loss / (batch_index + 1)

# Define the validation function
@torch.no_grad()
def validation(dataloader, gcn_model_name):
    total_loss = 0
    batch_index = 0
    model.eval()
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)
        with torch.no_grad():
            if gcn_model_name in ["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "SuperGATConv"]:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
            else:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            loss = loss_fn(output, y)
            total_loss += torch.sqrt(loss).cpu().detach().item()
        batch_index += 1
    return total_loss / (batch_index + 1)

def on_train(epochs_slider: gr.Slider, gcn_model_name: gr.Dropdown, progress=gr.Progress()):
    global trained_epochs
    epochs = epochs_slider
    t = progress.tqdm(range(trained_epochs+1, epochs+trained_epochs+1), total=epochs, desc="Training")
    for epoch in t:
        train_loss = train(train_dataloader, gcn_model_name)
        train_losses.append(train_loss)
        
        val_loss = validation(val_dataloader, gcn_model_name)
        val_losses.append(val_loss)
        
        scheduler.step()
        trained_epochs += 1

        # Plot the training loss
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
    file_path = f'.\\{dataset.dataset_name}_{trained_epochs}_losses.csv'
    df.to_csv(file_path)
    return f'Losses exported to {file_path}.'

# Define the test function
@torch.no_grad()
def test(dataloader, gcn_model_name):
    batch_index = 0
    model.eval()
    y_test = torch.tensor([])
    y_pred = torch.tensor([])
    smiles_arr = []
    for batch in dataloader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        y = batch.y.float().to(device)
        smiles = batch.smiles
        with torch.no_grad():
            if gcn_model_name in ["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "SuperGATConv"]:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
            else:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            y_test = torch.cat((y_test, y.cpu().detach()), dim=0)
            y_pred = torch.cat((y_pred, output.cpu().detach()), dim=0)
            smiles_arr.extend(smiles)
        batch_index += 1
    return y_test, y_pred, smiles_arr

def on_evaluate(gcn_model_name: gr.Dropdown):
    # Get the output scaler
    output_scaler = dataset.output_scaler
    
    # Run the model forward to get the test result
    global test_smiles_arr
    y_test_scaled, y_pred_scaled, test_smiles_arr = test(test_dataloader, gcn_model_name)

    # Scale the outputs back to the original range
    global y_test
    global y_pred
    y_test = output_scaler.inverse_transform(y_test_scaled.reshape(-1,1)).reshape(-1)
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)

    # Evaluate the model using different metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    html = """
    <div>
        <table>
            <thead>
                <tr>
                    <td><b>Metric</b></td>
                    <td><b>Value</b></td>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Mean absolute error</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>Mean squared error</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>Root mean squared error</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>Coefficient of determination</td>
                    <td>{:.4f}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """.format(mae, mse, rmse, r2)

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
    file_path = f'.\\{dataset.dataset_name}_{trained_epochs}_scatter_plot.csv'
    df.to_csv(file_path)
    return f'Scatter plot exported to {file_path}.'

def on_process_prediction_data(prediction_dataset_file: gr.File, prediction_dataset_name_textbox: gr.Textbox, gcn_featurizer_dropdown: gr.Dropdown, mol_featurizer_dropdown: gr.Dropdown, batch_size_dropdown: gr.Dropdown):
    # Load dataset
    dataset_file_path = prediction_dataset_file
    dataset_name = prediction_dataset_name_textbox
    gcn_featurizer_name = gcn_featurizer_dropdown
    mol_featurizer_name = mol_featurizer_dropdown
    
    # Process the dataset
    global prediction_dataset
    prediction_dataset = MoleculeDatasetForRegressionPrediction(dataset_file_path, dataset_name, gcn_featurizer_name, mol_featurizer_name)
    
    # DataLoaders
    global prediction_dataloader
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size_dropdown)

    return f'Data processed: {len(prediction_dataset)} molecules.'

# Define the predict function
@torch.no_grad()
def predict(dataloader, gcn_model_name):
    model.eval()
    smiles_list = []
    y_pred = torch.tensor([])
    for batch in dataloader:
        smiles_list.extend(batch.smiles)
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.long().to(device)
        mol_features = batch.mol_features.numpy()
        mol_features_scaled = mol_features_scaler.transform(mol_features)
        mol_features_scaled = torch.tensor(mol_features_scaled).float().to(device)
        with torch.no_grad():
            if gcn_model_name in ["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "SuperGATConv"]:
                output = model(x, edge_index, batch.batch.to(device), mol_features_scaled)
            else:
                edge_attr = batch.edge_attr.float().to(device)
                output = model(x, edge_index, edge_attr, batch.batch.to(device), mol_features_scaled)
            y_pred = torch.cat((y_pred, output.cpu().detach()), dim=0)
    return smiles_list, y_pred

def on_predict(gcn_model_name: gr.Dropdown):
    # Get the output scaler
    output_scaler = dataset.output_scaler
    
    # Run the model forward to get the test result
    smiles_list, y_pred_scaled = predict(prediction_dataloader, gcn_model_name)

    # Scale the outputs back to the original range
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    global prediction_df
    prediction_df = pd.DataFrame()
    prediction_df['smiles'] = smiles_list
    prediction_df[dataset.target_column] = y_pred.tolist()
    
    export_prediction_button = gr.Button(value="Export", interactive=True)

    return prediction_df, export_prediction_button

def on_export_prediction():
    file_path = f'.\\{prediction_dataset.dataset_name}_prediction.csv'
    prediction_df.to_csv(file_path)
    return f'Prediction exported to {file_path}.'

def gnn_regression_tab_content():
    with gr.Tab("GNN for Regression") as gnn_regression_tab:
        with gr.Accordion("Dataset"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Dataset file")
                    dataset_name_textbox = gr.Textbox(label="Dataset name", placeholder="dataset", value="dataset")
                    target_column_textbox = gr.Textbox(label="Target column")
                with gr.Column(scale=1):
                    gcn_featurizer_dropdown = gr.Dropdown(label="Graph featurizer", value="MolGraphConvFeaturizer", choices=["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"])
                    mol_featurizer_dropdown = gr.Dropdown(label="Molecule featurizer", value="Mordred descriptors", choices=["Mordred descriptors", "RDKit descriptors", "MACCS keys", "Morgan fingerprint", "Avalon fingerprint", "Atom-pairs fingerprint", "Topological-torsion fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint"])
                with gr.Column(scale=1):
                    test_size_slider = gr.Slider(minimum=0, maximum=0.4, value=0.2, step=0.01, label="Test size")
                    val_size_slider = gr.Slider(minimum=0, maximum=0.4, value=0.2, step=0.01, label="Validation size")
                    batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64])
                    random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Random seed")
                    process_data_button = gr.Button(value="Process data")
                    process_data_markdown = gr.Markdown()
        with gr.Accordion("GNN model"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Graph convolutional layers")
                        gcn_model_name = gr.Dropdown(label="Graph convolutional layer", value="GCN", choices=["GCN", "GraphSAGE", "CuGraphSAGE", "SGConv", "ClusterGCN", "GraphConv", "ChebConv", "LEConv", "EGConv", "MFConv", "FeaStConv", "TAGConv", "ARMAConv", "FiLMConv", "PDNConv", "GENConv", "ResGatedGraphConv", "GAT", "GATv2", "SuperGAT", "TransformerConv", "GeneralConv"])
                        gcn_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                        gcn_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                        gcn_n_heads = gr.Slider(label="n_heads", minimum=1, maximum=8, value=3, step=1)
                        gcn_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Molecular feature layers")
                        mlp_n_hiddens = gr.Dropdown(label="n_hiddens", value=128, choices=[16, 32, 64, 128, 256, 512])
                        mlp_n_layers = gr.Slider(label="n_layers", minimum=1, maximum=6, value=3, step=1)
                        mlp_n_outputs = gr.Slider(label="n_outputs", minimum=0, maximum=512, value=50, step=1)
                    with gr.Group():
                        gr.Markdown(value="Predictor layers")
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

    process_data_button.click(on_process_data, [dataset_file, dataset_name_textbox, target_column_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown, test_size_slider, val_size_slider, batch_size_dropdown, random_seed_slider],
                              [process_data_markdown, create_model_button])
    create_model_button.click(on_create_model, [gcn_model_name, gcn_n_hiddens, gcn_n_layers, gcn_n_heads, gcn_n_outputs,
                                                mlp_n_hiddens, mlp_n_layers, mlp_n_outputs,
                                                predictor_n_hiddens, predictor_n_layers, random_seed_slider],
                                                [model_summary_textarea, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button])
    save_checkpoint_button.click(on_save_checkpoint, model_name_textbox, save_checkpoint_markdown)
    load_checkpoint_button.click(on_load_checkpoint, checkpoint_file, load_checkpoint_markdown)
    create_optimizer_button.click(on_create_optimizer, [optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [create_optimizer_markdown, train_button])
    train_button.click(on_train, [epochs_slider, gcn_model_name], [training_plot, export_losses_button])
    export_losses_button.click(on_export_losses, [], export_losses_markdown)
    evaluate_button.click(on_evaluate, gcn_model_name, [evaluation_metrics_html, evaluation_plot, export_scatter_plot_button])
    export_scatter_plot_button.click(on_export_scatter_plot, [], export_scatter_plot_markdown)
    process_prediction_data_button.click(on_process_prediction_data, [prediction_dataset_file, prediction_dataset_name_textbox, gcn_featurizer_dropdown, mol_featurizer_dropdown, batch_size_dropdown], process_prediction_data_markdown)
    predict_button.click(on_predict, gcn_model_name, [prediction_datatable, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [], export_prediction_markdown)
    
    return gnn_regression_tab