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

Install [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

Install other packages

```
pip install rdkit
pip install mordred
pip install tensorflow
pip install deepchem
pip install gradio
pip install gradio_molecule2d
pip install gradio_molecule3d
pip install plotly
```

## Start web UI
To start the web UI:

```
conda activate ./gnn-env
set PYTHONUTF8=1
python webui.py
```