## Installation

- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gnn-webui
```

- Create and activate virtual environment:

```
cd gnn-webui
python -m venv gnn-env
gnn-env\Scripts\activate
```

- Install packages:

Install [PyTorch](https://pytorch.org/)

```

pip3 install torch --index-url https://download.pytorch.org/whl/cu128

```

Install [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

```

Install other packages

```
pip install rdkit
pip install mordred
pip install tensorflow
pip install ase
pip install deepchem
pip install gradio==5.29.1
pip install nglview
pip install plotly
pip install tabulate
```

## Start web UI
To start the web UI:

```
start_webui
```