import os
import gradio as gr
from gradio_molecule2d import molecule2d
from gradio_molecule3d import Molecule3D
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

def on_create_molecule(molecule_editor: molecule2d):
    file_path = '.\\molecule.pdb'
    try:
        mol = Chem.MolFromSmiles(molecule_editor)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        Chem.MolToPDBFile(mol, file_path)
    except:
        return None, None
    
    create_graph_button = gr.Button(value="Create graph", interactive=True)
    return file_path, create_graph_button

def on_create_graph(smiles: gr.Textbox, featurizer_name: gr.Dropdown):
    # Extract the SMILES string from the event
    if smiles == '':
        return [None, None, None, None, None]
    
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Check if the molecule is valid
    if molecule is None:
        return [None, None, None, None, None]

    if (featurizer_name == "MolGraphConvFeaturizer"):
        featurizer = dc.deepchem.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
    elif (featurizer_name == "PagtnMolGraphFeaturizer"):
        featurizer = dc.deepchem.feat.PagtnMolGraphFeaturizer(max_length=5)
    elif (featurizer_name == "DMPNNFeaturizer"):
        featurizer = dc.deepchem.feat.DMPNNFeaturizer(is_adding_hs=False)
    else:
        return [None, None, None, None, None]
    
    features = featurizer.featurize(molecule)     
    node_feats = np.array(features[0].node_features)
    edge_index = np.array(features[0].edge_index)
    edge_attr = np.array(features[0].edge_features)

    # Create a networkx graph
    G = nx.Graph()
    num_nodes = node_feats.shape[0]
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T.tolist())

    # Draw the graph
    molecule_graph = draw_interactive_graph(G, node_feats)

    return [molecule_graph, node_feats, edge_index.T, edge_index_to_adj(edge_index, num_nodes), edge_attr]

def draw_interactive_graph(G, node_feats):
    # Get node positions using networkx spring layout
    pos = nx.spring_layout(G)
    
    # Extract position components
    Xn = [pos[k][0] for k in range(node_feats.shape[0])]
    Yn = [pos[k][1] for k in range(node_feats.shape[0])]

    # Prepare DataFrame for edges
    edge_data = {
        'x': [],
        'y': [],
    }

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data['x'].extend([x0, x1, None])
        edge_data['y'].extend([y0, y1, None])

    edge_df = pd.DataFrame(edge_data)
    
    # Create edges using plotly.express.line with gray color
    edge_fig = px.line(
        edge_df,
        x='x', 
        y='y', 
        color_discrete_sequence=['gray']  # Set edges to gray
    )

    # Create DataFrame for nodes
    node_data = pd.DataFrame({
        'x': Xn,
        'y': Yn,
        'size': [30] * len(Xn)
    })

    # Create nodes using plotly.express.scatter with blue color and atom type hover text
    node_fig = px.scatter(
        node_data,
        x='x',
        y='y',
        size='size',
        color_discrete_sequence=['blue'],  # Set nodes to blue
        hover_data={'x': False, 'y': False, 'size': False}  # Hide x, y, and size data from hover
    )

    # Node labels
    node_labels = go.Scatter(
        x=Xn,
        y=Yn,
        mode='text',
        text=[str(i) for i in range(len(Xn))],  # Assuming nodes have sequential ids
        textfont=dict(color='white', size=12),  # White font color for labels
        hoverinfo='none'  # No hover info for labels
    )

    # Combine the plots into one figure
    fig = go.Figure()

    for trace in edge_fig.data:
        fig.add_trace(trace)

    for trace in node_fig.data:
        fig.add_trace(trace)
    
    fig.add_trace(node_labels) 

    # Adjust the layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig

def edge_index_to_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    return adj

reps = [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "stick",
      "color": "whiteCarbon",
      "residue_range": "",
      "around": 0,
      "byres": False,
      "visible": False
    }
]

def graph_tab_content():
    with gr.Tab("Graph") as graph_tab:
        with gr.Accordion("Molecule"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    molecule_editor = molecule2d(label="SMILES")
                with gr.Column(scale=1):
                    create_molecule_button = gr.Button(value="Create molecule")
                    molecule_viewer = Molecule3D(label="Molecule" , reps=reps)
        with gr.Accordion("Graph representation"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    featurizer_dropdown = gr.Dropdown(label="Featurizer", value="MolGraphConvFeaturizer", choices=["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"])
                with gr.Column(scale=2):
                    create_graph_button = gr.Button(value="Create graph", interactive=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    graph_plot = gr.Plot()
                with gr.Column(scale=2):
                    adj_matrix_datatable = gr.DataFrame(label="Adjacency matrix", wrap=False, interactive=False)
            with gr.Row():
                node_features_datatable = gr.DataFrame(label="Node features", wrap=False, interactive=False)
            with gr.Row():
                with gr.Column(scale=1):
                    edge_index_datatable = gr.DataFrame(label="Edge index", wrap=False, interactive=False)
                with gr.Column(scale=5):
                    edge_features_datatable = gr.DataFrame(label="Edge features", wrap=False, interactive=False)
    
        create_molecule_button.click(on_create_molecule, molecule_editor, [molecule_viewer, create_graph_button])
        create_graph_button.click(on_create_graph, [molecule_editor, featurizer_dropdown], [graph_plot, node_features_datatable, edge_index_datatable, adj_matrix_datatable, edge_features_datatable])
    
    return graph_tab