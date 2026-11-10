## Introduction

A Gradio web app for molecular property prediction with hybrid and 3D graph neural
networks (GNNs), and classical scikit-learn models trained on molecular fingerprints.

The six workflows are:

| Tab | Module | Features |
| --- | --- | --- |
| Binary Classification - Hybrid GNN | `binary_classification_hybrid.py` | Graphs and/or fingerprints |
| Binary Classification - 3D GNN | `binary_classification_3d.py` | 3D conformers |
| Binary Classification - Classical ML Models | `binary_classification_classical.py` | Fingerprints |
| Regression - Hybrid GNN | `regression_hybrid.py` | Graphs and/or fingerprints |
| Regression - 3D GNN | `regression_3d.py` | 3D conformers |
| Regression - Classical ML Models | `regression_classical.py` | Fingerprints |

The neural-network block names in the interface and model summaries are
**Graph Encoder**, **Fingerprint Encoder**, and **Prediction Head**. The graph and
fingerprint encoders produce molecular representations; the prediction head transforms
the selected encoder outputs before the final output layer. Algorithm names such as
GCN, GraphSAGE, SchNet, and the classical Multilayer Perceptron remain unchanged.


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

## Using the app

1. Choose a classification or regression tab and create or open a working directory.
2. Upload a CSV, identify its SMILES and target columns, and click **Load dataset**.
   SMILES cleanup options are enabled by default.
3. Extract the features your model needs: graphs and/or fingerprints for hybrid
   GNNs, conformers for 3D GNNs, or fingerprints for classical ML.
4. Choose **Random** or **Scaffold** splitting, set the data split seed, and process
   the data. Classical ML uses a 20% test split by default.
5. Choose the model and its parameters, name the run, and set its model seed. Create
   and train the GNN model, or fit the classical model.
6. Evaluate the model, save its checkpoint, and export metrics or predictions.
   Use **Make predictions** for new molecules.

Working directories live under `Data/Classification/` or `Data/Regression/`.
Graphs, fingerprints, and model results are stored in `GRAPH_`, `FP_`, and `MODEL_`
folders. Classical tabs can also extract graphs, while their models use fingerprints.

## More details

- [User guide](docs/user-guide.md): dataset options, feature extraction, splits,
  similarity, model choices, checkpoints, and exports.
- [Development and troubleshooting](docs/development.md): shared modules, tests,
  and runtime diagnostics.
