import os
import math
import pandas as pd
import torch
import gradio as gr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus, ViSNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from datasets import *
from models import *

def on_process_data(dataset_file: gr.File, dataset_name_textbox: gr.Textbox, target_column_textbox: gr.Textbox, num_conformers_slider:gr.Slider, test_size_slider: gr.Slider, val_size_slider: gr.Slider, batch_size_dropdown: gr.Dropdown, random_seed_slider: gr.Slider):
    # Load dataset
    dataset_file_path = dataset_file
    dataset_name = dataset_name_textbox
    target_column = target_column_textbox

    try:
        # Process the dataset
        global dataset
        dataset = MoleculeDatasetForRegression3D(dataset_file_path, dataset_name, target_column, num_conformers_slider)
        
        # Train-validation-test splitting
        test_size = int(len(dataset) * test_size_slider)
        val_size = int(len(dataset) * val_size_slider)
        train_size = len(dataset) - test_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed_slider))

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
    return f"Data processed: {len(dataset)} molecules.",create_model_button

gnn_model_tab = "SchNet"
def on_gnn_model_tab_selected(evt: gr.SelectData):
    if evt.selected: 
        global gnn_model_tab
        gnn_model_tab = evt.value
    
    return

def on_create_model(schnet_n_hiddens: gr.Dropdown, schnet_n_filters: gr.Dropdown, schnet_n_interactions: gr.Slider, schnet_n_gaussians: gr.Slider, schnet_cutoff: gr.Slider, schnet_max_num_neighbors: gr.Slider, schnet_readout: gr.Radio, schnet_dipole: gr.Checkbox,
                    dimenetplusplus_n_hiddens: gr.Dropdown, dimenetplusplus_n_blocks: gr.Slider, dimenetplusplus_int_emb_size: gr.Dropdown, dimenetplusplus_basis_emb_size: gr.Dropdown, dimenetplusplus_out_emb_channels: gr.Dropdown, dimenetplusplus_n_spherical: gr.Slider, dimenetplusplus_n_radial: gr.Slider, dimenetplusplus_cutoff: gr.Slider, dimenetplusplus_max_num_neighbors: gr.Slider, dimenetplusplus_envelope_exponent: gr.Slider, dimenetplusplus_n_output_layers: gr.Slider,
                    visnet_n_hiddens: gr.Dropdown, visnet_n_heads: gr.Slider, visnet_n_layers: gr.Slider, visnet_num_rbf: gr.Slider, visnet_trainable_rbf: gr.Checkbox, visnet_max_z: gr.Slider, visnet_cutoff: gr.Slider, visnet_max_num_neighbors: gr.Slider, visnet_vertex: gr.Checkbox, random_seed_slider: gr.Slider):
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_seed_slider)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed_slider)

    global model

    if gnn_model_tab == "SchNet":
        model = SchNet(hidden_channels=schnet_n_hiddens, num_filters=schnet_n_filters, num_interactions=schnet_n_interactions, 
               num_gaussians=schnet_n_gaussians, cutoff=schnet_cutoff, max_num_neighbors=schnet_max_num_neighbors,
               readout=schnet_readout, dipole=schnet_dipole).float().to(device)
    elif gnn_model_tab == "DimeNet++":
        model = DimeNetPlusPlus(hidden_channels=dimenetplusplus_n_hiddens, out_channels=1, num_blocks=dimenetplusplus_n_blocks,
                        int_emb_size=dimenetplusplus_int_emb_size, basis_emb_size=dimenetplusplus_basis_emb_size,
                        out_emb_channels=dimenetplusplus_out_emb_channels, num_spherical=dimenetplusplus_n_spherical,
                        num_radial=dimenetplusplus_n_radial, cutoff=dimenetplusplus_cutoff,
                        max_num_neighbors=dimenetplusplus_max_num_neighbors, envelope_exponent=dimenetplusplus_envelope_exponent,
                        num_output_layers=dimenetplusplus_n_output_layers).float().to(device)
    elif gnn_model_tab == "ViSNet":
        model = ViSNet(num_heads=visnet_n_heads, num_layers=visnet_n_layers, hidden_channels=visnet_n_hiddens, num_rbf=visnet_num_rbf,
                       trainable_rbf=visnet_trainable_rbf, max_z=visnet_max_z, cutoff=visnet_cutoff, max_num_neighbors=visnet_max_num_neighbors,
                       vertex=visnet_vertex).float().to(device)
    
    global train_losses
    global val_losses
    global trained_epochs
    train_losses = []
    val_losses = []
    trained_epochs = 0

    save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=True)
    load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=True)
    create_optimizer_button = gr.Button(value="Create optimizer", interactive=True)
    evaluate_button = gr.Button(value="Evaluate", interactive=True)
    predict_button = gr.Button(value="Predict", interactive=True)

    return gnn_model_tab, f'Model created: {gnn_model_tab}.', save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button

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
def train(dataloader, model_name):
    total_loss = 0
    batch_index = 0
    model.train()
    for batch in dataloader:
        z = batch.z.long().to(device)
        pos = batch.pos.float().to(device)
        y = batch.y.float().to(device)
        if model_name == "SchNet" or model_name == "DimeNet++":
            output = model(z, pos, batch.batch.to(device))
        else: # nn_model_name == "ViSNet"
            output, _ = model(z, pos, batch.batch.to(device))
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += torch.sqrt(loss).cpu().detach().item()
        batch_index += 1
    return total_loss / (batch_index + 1)

# Define the validation function
@torch.no_grad()
def validation(dataloader, model_name):
    total_loss = 0
    batch_index = 0
    model.eval()
    for batch in dataloader:
        z = batch.z.long().to(device)
        pos = batch.pos.float().to(device)
        y = batch.y.float().to(device)
        with torch.no_grad():
            if model_name == "SchNet" or model_name == "DimeNet++":
                output = model(z, pos, batch.batch.to(device))
            else: # nn_model_name == "ViSNet"
                output, _ = model(z, pos, batch.batch.to(device))
            loss = loss_fn(output, y)
            total_loss += torch.sqrt(loss).cpu().detach().item()
        batch_index += 1
    return total_loss / (batch_index + 1)

def on_train(epochs_slider: gr.Slider, created_model_name: gr.Textbox, progress=gr.Progress()):
    global trained_epochs
    epochs = epochs_slider
    t = progress.tqdm(range(trained_epochs+1, epochs+trained_epochs+1), total=epochs, desc="Training")
    for epoch in t:
        train_loss = train(train_dataloader, created_model_name)
        train_losses.append(train_loss)
        
        val_loss = validation(val_dataloader, created_model_name)
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
    file_path = f'./{dataset.dataset_name}_{trained_epochs}_losses.csv'
    df.to_csv(file_path)
    return f'Losses exported to {file_path}.'

# Define the test function
@torch.no_grad()
def test(dataloader, model_name):
    batch_index = 0
    model.eval()
    y_test = torch.tensor([])
    y_pred = torch.tensor([])
    smiles_arr = []
    for batch in dataloader:
        z = batch.z.long().to(device)
        pos = batch.pos.float().to(device)
        y = batch.y.float().to(device)
        smiles = batch.smiles
        with torch.no_grad():
            if model_name == "SchNet" or model_name == "DimeNet++":
                output = model(z, pos, batch.batch.to(device))
            else: # nn_model_name == "ViSNet"
                output, _ = model(z, pos, batch.batch.to(device))
            y_test = torch.cat((y_test, y.cpu().detach()), dim=0)
            y_pred = torch.cat((y_pred, output.cpu().detach()), dim=0)
            smiles_arr.extend(smiles)
        batch_index += 1
    return y_test, y_pred, smiles_arr

def on_evaluate(created_model_name: gr.Textbox):
    # Get the output scaler
    output_scaler = dataset.output_scaler
    
    # Run the model forward to get the test result
    global test_smiles_arr
    y_test_scaled, y_pred_scaled, test_smiles_arr = test(test_dataloader, created_model_name)

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
    file_path = f'./{dataset.dataset_name}_{trained_epochs}_scatter_plot.csv'
    df.to_csv(file_path)
    return f'Scatter plot exported to {file_path}.'

def on_process_prediction_data(prediction_dataset_file: gr.File, prediction_dataset_name_textbox: gr.Textbox, num_conformers_slider: gr.Slider, batch_size_dropdown: gr.Dropdown):
    # Load dataset
    dataset_file_path = prediction_dataset_file
    dataset_name = prediction_dataset_name_textbox
    
    # Process the dataset
    global prediction_dataset
    prediction_dataset = MoleculeDatasetForRegressionPrediction3D(dataset_file_path, dataset_name, num_conformers_slider)
    
    # DataLoaders
    global prediction_dataloader
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size_dropdown)

    return f'Data processed: {len(prediction_dataset)} molecules.'

# Define the predict function
@torch.no_grad()
def predict(dataloader, model_name):
    model.eval()
    smiles_list = []
    y_pred = torch.tensor([])
    for batch in dataloader:
        smiles_list.extend(batch.smiles)
        z = batch.z.long().to(device)
        pos = batch.pos.float().to(device)
        with torch.no_grad():
            if model_name == "SchNet" or model_name == "DimeNet++":
                output = model(z, pos, batch.batch.to(device))
            else: # nn_model_name == "ViSNet"
                output, _ = model(z, pos, batch.batch.to(device))
            y_pred = torch.cat((y_pred, output.cpu().detach()), dim=0)

    return smiles_list, y_pred

def on_predict(created_model_name: gr.Textbox):
    # Get the output scaler
    output_scaler = dataset.output_scaler
    
    # Run the model forward to get the test result
    smiles_list, y_pred_scaled = predict(prediction_dataloader, created_model_name)

    # Scale the outputs back to the original range
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    global prediction_df
    prediction_df = pd.DataFrame()
    prediction_df['smiles'] = smiles_list
    prediction_df[dataset.target_column] = y_pred.tolist()
    
    export_prediction_button = gr.Button(value="Export", interactive=True)

    return prediction_df, export_prediction_button

def on_export_prediction():
    file_path = f'./{prediction_dataset.dataset_name}_prediction.csv'
    prediction_df.to_csv(file_path)
    return f'Prediction exported to {file_path}.'

def gnn_regression_3d_tab_content():
    with gr.Tab("3D GNN for Regression") as gnn_regression_3d_tab:
        with gr.Accordion("Dataset"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    dataset_file = gr.File(file_types=['.csv'], type='filepath', label="Dataset file")
                    dataset_name_textbox = gr.Textbox(label="Dataset name", placeholder="dataset", value="dataset")
                    target_column_textbox = gr.Textbox(label="Target column")
                    num_conformers_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of conformers")
                with gr.Column(scale=1):
                    test_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Test size")
                    val_size_slider = gr.Slider(minimum=0, maximum=1.0, value=0.2, step=0.01, label="Validation size")
                    batch_size_dropdown = gr.Dropdown(label="Batch size", value=32, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
                    random_seed_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Random seed")
                    process_data_button = gr.Button(value="Process data")
                    process_data_markdown = gr.Markdown()
        with gr.Accordion("3D GNN model"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
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
                            schnet_tab.select(on_gnn_model_tab_selected, [], [])
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
                            dimenetplusplus_tab.select(on_gnn_model_tab_selected, [], [])
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
                            visnet_tab.select(on_gnn_model_tab_selected, [], [])
                with gr.Column(scale=1):
                    create_model_button = gr.Button(value="Create model", interactive=False)
                    created_model_name_textbox = gr.Text(value="SchNet", visible=False)
                    model_summary_markdown = gr.Markdown()
                    model_name_textbox = gr.Textbox(label="Model name", placeholder="model", value="model")
                    save_checkpoint_button = gr.Button(value="Save checkpoint", interactive=False)
                    save_checkpoint_markdown = gr.Markdown()
                    checkpoint_file = gr.File(file_types=['.ckpt'], type='filepath', label="Checkpoint file")
                    load_checkpoint_button = gr.Button(value="Load checkpoint", interactive=False)
                    load_checkpoint_markdown = gr.Markdown()
        with gr.Accordion("Model training"):
            with gr.Row(equal_height=True): 
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(value="Optimizer")
                        optimizer_dropdown = gr.Dropdown(label="Optimizer", value="Adam", choices=["Adam", "AdamW", "SGD"]) 
                        learning_rate_slider = gr.Slider(label="Learning rate", minimum=0.00001, maximum=0.1, value=0.0001, step=0.00001) 
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

    process_data_button.click(on_process_data, [dataset_file, dataset_name_textbox, target_column_textbox, num_conformers_slider, test_size_slider, val_size_slider, batch_size_dropdown, random_seed_slider],
                              [process_data_markdown, create_model_button])
    create_model_button.click(on_create_model, [schnet_n_hiddens, schnet_n_filters, schnet_n_interactions, schnet_n_gaussians, schnet_cutoff, schnet_max_num_neighbors, schnet_readout, schnet_dipole,
                                                dimenetplusplus_n_hiddens, dimenetplusplus_n_blocks, dimenetplusplus_int_emb_size, dimenetplusplus_basis_emb_size, dimenetplusplus_out_emb_channels, dimenetplusplus_n_spherical, dimenetplusplus_n_radial, dimenetplusplus_cutoff, dimenetplusplus_max_num_neighbors, dimenetplusplus_envelope_exponent, dimenetplusplus_n_output_layers,
                                                visnet_n_hiddens, visnet_n_heads, visnet_n_layers, visnet_num_rbf, visnet_trainable_rbf, visnet_max_z, visnet_cutoff, visnet_max_num_neighbors, visnet_vertex, random_seed_slider],
                                                [created_model_name_textbox, model_summary_markdown, save_checkpoint_button, load_checkpoint_button, create_optimizer_button, evaluate_button, predict_button])
    save_checkpoint_button.click(on_save_checkpoint, model_name_textbox, save_checkpoint_markdown)
    load_checkpoint_button.click(on_load_checkpoint, checkpoint_file, load_checkpoint_markdown)
    create_optimizer_button.click(on_create_optimizer, [optimizer_dropdown, learning_rate_slider, learning_rate_decay_slider], [create_optimizer_markdown, train_button])
    train_button.click(on_train, [epochs_slider, created_model_name_textbox], [training_plot, export_losses_button])
    export_losses_button.click(on_export_losses, [], export_losses_markdown)
    evaluate_button.click(on_evaluate, created_model_name_textbox, [evaluation_metrics_html, evaluation_plot, export_scatter_plot_button])
    export_scatter_plot_button.click(on_export_scatter_plot, [], export_scatter_plot_markdown)
    process_prediction_data_button.click(on_process_prediction_data, [prediction_dataset_file, prediction_dataset_name_textbox, num_conformers_slider, batch_size_dropdown], process_prediction_data_markdown)
    predict_button.click(on_predict, created_model_name_textbox, [prediction_datatable, export_prediction_button])
    export_prediction_button.click(on_export_prediction, [], export_prediction_markdown)
    
    return gnn_regression_3d_tab