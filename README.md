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

## Working directories

Classification tabs use `Data/Classification/` as their working-directory base;
regression tabs use `Data/Regression/`. The **Working Directory** selector lists
folders within the corresponding base. Select a folder or enter a new name, then
click **Create/Open Working Directory**. Hybrid, 3D, and classical tabs share their
task's base.

The example datasets are organized as follows:

```text
Data/
  Classification/
    MTX/
    MTX_3D/
  Regression/
    ESOL/
    ESOL_3D/
```

Each dataset folder holds its CSV files, extracted features, and checkpoints.

## Data splitting and evaluation

Each tab offers **Split method** radio buttons in **Load and process data**:

- **Random** (default) keeps the existing seeded random split.
- **Scaffold** groups molecules by their [RDKit Murcko scaffold](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
  after the same salt removal used for feature extraction. Molecules sharing a scaffold
  stay together, with stereochemistry ignored for grouping. All acyclic molecules share
  the empty scaffold. Larger groups are assigned first, and the random seed breaks ties.

Classical ML workflows use training and test sets only, with a default test size of
20%. All remaining rows are used for training. GNN workflows retain separate
training, validation, and test sets.

Scaffold split sizes are approximate: the status message reports the actual counts
in each partition. If there are too few distinct scaffolds to populate the
requested nonempty splits, processing reports an error; reduce the holdout sizes or
choose Random. A holdout fraction that rounds down to zero rows remains empty.
Preprocessing is fitted only on the training split with either method. The GNN tabs
also support PCA, variance filtering, and regression target scaling; shuffling their
training batches is separate from choosing a split. Processing data clears the current
model and results, along with any GNN optimizer and training history. Create or fit a
model again for the new split, so a model trained on an older split cannot be evaluated
against a new holdout accidentally.

All classification tabs report accuracy, precision, recall, specificity, F1,
ROC-AUC, balanced accuracy, MCC, AUPRC, and average precision. Hybrid and classical
workflows support multiple target columns, with metrics per target. AUPRC uses trapezoidal integration of the
precision–recall curve; [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
is reported separately because it uses recall-weighted precision without interpolation.

For a test split containing only one class, the app shows `NaN` for ROC-AUC, AUPRC,
average precision, balanced accuracy, and MCC instead of presenting them as measures
of discrimination. Recall is unavailable without positive labels, and specificity is
unavailable without negative labels. Precision and F1 use zero for zero denominators;
MCC is zero for constant predictions when both true classes are present. Regression
metrics are MAE, MSE, RMSE, and R², in the original target units. R² is unavailable
for a single test row or constant actual values.

### Export evaluation results

All six tabs provide two exports in **Model evaluation**:

| File field | Button | CSV contents |
| --- | --- | --- |
| **Metrics export file** (default `test_metrics`) | **Export test metrics (all targets)** | One row per target and metric: `Target`, `Metric`, `Value` |
| **Evaluation export file** (default `evaluation`) | **Export evaluation rows (all targets)** | One row per test molecule: `SMILES`, actual target values, `<target> (Prediction)`, and classification `<target> (Probability)` |

Evaluate the model first, then choose a filename and click the corresponding export
button. Files are saved in the currently opened working directory. Names may include
the `.csv` extension; it is added if omitted. Exporting to an existing filename replaces
that file. Undefined metrics are written as empty CSV cells.

Both exports include **all targets**, even when another output is selected for the
displayed metrics and plots. They reuse the completed evaluation without rerunning
prediction. Regression CSV metrics retain full precision; GNN metric tables display
four decimal places. Processing a new split or changing/training/loading the model
clears old results and disables their exports; evaluate or predict again afterward.

## Classical ML workflows

Start with **Random Forest**, the default in both classical tabs, then compare it
with a linear baseline and the other models. The best choice depends on the dataset.

| Classification | Regression | When to try it |
| --- | --- | --- |
| Random Forest | Random Forest | A first nonlinear baseline with little parameter tuning |
| Extra Trees | Extra Trees | An alternative tree ensemble with more randomized splits |
| Gradient Boosting | Gradient Boosting | Sequential trees; tune tree depth, learning rate, and number of trees |
| Logistic Regression | Ridge | A regularized linear baseline |
| SVC | SVR | A kernel model; tune C, kernel, and gamma |
| K-Nearest Neighbors | K-Nearest Neighbors | A local similarity baseline; tune neighbor count and weighting |

These starting points are scikit-learn's [tree ensembles](https://scikit-learn.org/stable/modules/ensemble.html),
[linear models](https://scikit-learn.org/stable/modules/linear_model.html),
[support vector machines](https://scikit-learn.org/stable/modules/svm.html), and
[nearest-neighbor models](https://scikit-learn.org/stable/modules/neighbors.html).
The model dropdown occupies the left column (scale 1); the wider right column
(scale 2) shows only the parameter controls relevant to the selected model.

The complete catalogue contains **24 classifiers and 29 regressors**:

| Family | Classification models | Regression models |
| --- | --- | --- |
| Trees and ensembles | Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, AdaBoost, Bagging, Decision Tree, Extra Tree | Same eight model families |
| Linear and robust models | Logistic Regression, SGD Classifier, Ridge Classifier, Perceptron | Linear Regression, Ridge, Lasso, Elastic Net, Bayesian Ridge, ARD Regression, Huber Regression, SGD Regressor, Quantile Regression, RANSAC Regression, Orthogonal Matching Pursuit |
| SVM and kernel ridge | SVC, Linear SVC | SVR, Linear SVR, Kernel Ridge |
| Naive Bayes | Gaussian, Bernoulli, Multinomial, Complement | — |
| Discriminant analysis | Linear Discriminant Analysis, Quadratic Discriminant Analysis | — |
| Generalized linear models | — | Poisson Regression, Gamma Regression, Tweedie Regression |
| Neighbors | K-Nearest Neighbors | K-Nearest Neighbors |
| Gaussian processes | Gaussian Process | Gaussian Process |
| Neural networks | Multilayer Perceptron | Multilayer Perceptron |
| Reference baselines | Dummy Classifier | Dummy Regressor |

Model-specific guidance appears with the parameter controls. SVC, Linear SVC, Ridge
Classifier, and Perceptron use sigmoid probability calibration within the training
split, including feature scaling separately within each calibration fold. They need
at least two training examples of each class per target. Poisson targets must be
nonnegative with at least one positive value; Gamma targets must be strictly
positive; Tweedie's requirements depend on the chosen power. Gaussian processes,
kernel ridge, and some wide linear models can be expensive on large datasets.

Open a working folder, upload a CSV with SMILES and one or more target columns,
and extract fingerprints. Existing aligned fingerprint folders can also be used.
Process the data with Random or Scaffold splitting, select a model, and train.
The estimator and any feature scaling are fitted on the training partition only.
Classification targets must be 0/1, with both classes present in each training
target. Regression predictions and metrics use the original target units.

After fitting, use **Model evaluation** to evaluate the test set. Select an output
to view its metrics and plots: classification shows a confusion matrix and ROC
curve, while regression shows actual versus predicted values with an identity line.
The plots share the right column, with evaluation controls and metrics on the left.
A single-class test set shows an explanation in place of its undefined ROC curve.
The **Model evaluation** and **Make predictions** accordions use consistent titles
and column layouts across all six tabs.

Save fitted models as `.joblib` files to retain their preprocessing,
target names, feature schema, and split identity. A loaded model can predict on new
compatible fingerprints; holdout evaluation requires the same processed dataset
and split used to train it. Only load model files you trust, as joblib uses pickle.

For prediction-only CSVs, leave the target-column field empty when uploading.
Extract prediction fingerprints with the same settings used for training, then
predict and export the results. New fingerprint extractions record their algorithm,
radius, and bit width in `fingerprint_config.json`. If an old extraction has no
settings metadata and cannot be matched to a newer one, re-extract both with the
same settings before training and prediction.
When both folders lack metadata, the app can check column names and width but
cannot verify the fingerprint algorithm. Re-extract them if their settings are uncertain.

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
pytest tests/test_classical_api.py tests/test_evaluation_exports_api.py  # all six export flows through Gradio
pytest -k checkpoint                 # one topic across all files
```

The API tests exercise Gradio preprocessing, session state, and plot serialization;
they require an environment that allows local sockets.

If Python segfaults during Gradio imports, before application code runs, the following
allocator workaround allowed the complete suite to pass in the local environment:

```bash
PYTHONMALLOC=malloc python -m pytest
```

This uses the system memory allocator instead of Python's small-object allocator.
The underlying native crash remains undiagnosed; no dependency versions were changed
to work around it. Keep the crash log when investigating the environment.
