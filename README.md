## Introduction

A Gradio web app for molecular property prediction with graph neural networks (GNNs) and molecular fingerprints.


## Installation

- Install [Python 3.11](https://www.python.org/downloads/) (other versions may not be compatible):
  
- Install [Git](https://git-scm.com/)

- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gnn-webui
```

- Create and activate virtual environment:

**Windows**
```
cd gnn-webui
python -m venv gnn-env
gnn-env\Scripts\activate
```

**Linux**
```
cd gnn-webui
python3 -m venv gnn-env
source gnn-env/bin/activate
```

- Install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive) (newer versions are not supported yet):
  

- Install packages:

Install [PyTorch](https://pytorch.org/) with CUDA version 12.8

```
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Install other packages:

```
pip install -r requirements.txt

pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.9.1+cu128.html

```

## Start web UI
To start the web UI:

```
python webui.py
```