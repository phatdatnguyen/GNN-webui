import os
import glob
import gradio as gr
from regression import regression_tab_content
from binary_classification import binary_classification_tab_content
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket

# remove all html files in the static directory
for filepath in glob.iglob('./Static/*.html'):
    os.remove(filepath)

# create a static directory to store the static files
static_dir = Path('./Static')
static_dir.mkdir(parents=True, exist_ok=True)

# create a data directory to store the user data files
data_dir = Path('./Data')
data_dir.mkdir(parents=True, exist_ok=True)

# create a FastAPI app
app = FastAPI()

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

with gr.Blocks() as blocks:
    with gr.Tabs() as tabs:
        binary_classification_tab_content()
        regression_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, css_paths=Path('./styles.css'), path="/", allowed_paths=["./", "./Static"])

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port, access_log=False)
