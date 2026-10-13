import gradio as gr
from binary_classification_hybrid import binary_classification_hybrid_tab_content
from binary_classification_3d import binary_classification_3d_tab_content
from binary_classification_classical import binary_classification_classical_tab_content
from regression_hybrid import regression_hybrid_tab_content
from regression_3d import regression_3d_tab_content
from regression_classical import regression_classical_tab_content
from pathlib import Path
from fastapi import FastAPI
from utils import BASE_DIR, DATA_DIR
import uvicorn
import socket
import os

# create a data directory to store the user data files
os.makedirs(DATA_DIR, exist_ok=True)

# create a FastAPI app
app = FastAPI()

# function to reserve an available port
def reserve_available_port(start_port=7860, max_attempts=100):
    """Bind and return a listening socket, so nothing else can take the port before uvicorn starts."""
    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.listen(128)
            return sock, port
        except OSError:
            sock.close()
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts - 1}.")

with gr.Blocks() as blocks:
    with gr.Tabs() as tabs:
        binary_classification_hybrid_tab_content()
        binary_classification_3d_tab_content()
        binary_classification_classical_tab_content()
        regression_hybrid_tab_content()
        regression_3d_tab_content()
        regression_classical_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, css_paths=Path(BASE_DIR) / 'styles.css', path="/")

# serve the app
if __name__ == "__main__":
    server_socket, available_port = reserve_available_port()
    # flush: stdout is block-buffered when redirected to a file, which would otherwise hide
    # this line until the process exits.
    print(f"Starting GNN web UI on http://127.0.0.1:{available_port}", flush=True)
    config = uvicorn.Config(app, access_log=False)
    uvicorn.Server(config).run(sockets=[server_socket])
