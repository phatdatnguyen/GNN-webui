## Installation
You will need [Anaconda](https://www.anaconda.com/download) for this app.
- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gnn-webui
```

- Create and activate Anaconda environment:

```
cd gnn-webui
conda create -p ./gnn-env
conda activate ./gnn-env
```

- Install packages:

Install [PyTorch](https://pytorch.org/)

```
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
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
pip install gradio
pip install gradio_molecule2d
pip install gradio_molecule3d
pip install plotly
pip install tabulate
```

## Start web UI
To start the web UI:

```
conda activate ./gnn-env
set PYTHONUTF8=1
python webui.py
```