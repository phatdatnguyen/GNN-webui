## Introduction

A Gradio web app for molecular property prediction with graph neural networks (GNNs) and molecular fingerprints.


## Installation

- Install [Python 3.11](https://www.python.org/downloads/) (other versions may not be compatible)

- `deepchem` 2.8.0 or newer is required: the `PagtnMolGraphFeaturizer` and `DMPNNFeaturizer` options
  do not exist in older releases.
  
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

**Linux**
```
cd gnn-webui
python3 -m venv gnn-env
source gnn-env/bin/activate
```
- Install packages:

Install [PyTorch](https://pytorch.org/) with CUDA version 13.2

```
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
```

Install other packages:

```
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cu132.html

```
## Start web UI
To start the web UI:

```
python webui.py
```

## Tests

Install the test dependencies, then run the suite from the repo root:

```
pip install -r requirements-dev.txt
pytest
```

The 3D tests need conformer generation, which is the slow part. To skip them:

```
pytest -m "not slow"
```

Useful selections:

```
pytest tests/test_models.py          # every architecture, at every layer count
pytest tests/test_handlers.py        # the Gradio callbacks end to end
pytest -k checkpoint                 # one topic across all files
```