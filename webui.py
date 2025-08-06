import os
import glob
import gradio as gr
from graph import graph_tab_content
from gnn_regression_hybrid import hybrid_gnn_regression_tab_content
from gnn_regression_3d import gnn_regression_3d_tab_content
from gnn_binary_classification_hybrid import hybrid_gnn_binary_classification_tab_content
from gnn_binary_classification_3d import gnn_binary_classification_3d_tab_content
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket

for filepath in glob.iglob('./Static/*.html'):
    os.remove(filepath)

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./Static')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/Static", StaticFiles(directory=static_dir), name="Static")

# function to find an available port
def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port

available_port = find_available_port()

with gr.Blocks(css='./styles.css') as blocks:
    with gr.Tabs() as tabs:
        graph_tab_content()
        hybrid_gnn_regression_tab_content()
        gnn_regression_3d_tab_content()
        hybrid_gnn_binary_classification_tab_content()
        gnn_binary_classification_3d_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, path="/", allowed_paths=["./", "./Static"])

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port)
