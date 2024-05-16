import gradio as gr
from graph import graph_tab_content
from gnn_regression import gnn_regression_tab_content
from gnn_regression_3d import gnn_regression_3d_tab_content
from gnn_binary_classification import gnn_binary_classification_tab_content
from gnn_binary_classification_3d import gnn_binary_classification_3d_tab_content

with gr.Blocks(css='styles.css') as app:
    with gr.Tabs() as tabs:
        graph_tab_content()
        gnn_regression_tab_content()
        gnn_regression_3d_tab_content()
        gnn_binary_classification_tab_content()
        gnn_binary_classification_3d_tab_content()

app.launch(share=True, allowed_paths=[".\\"])
