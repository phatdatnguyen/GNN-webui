# User guide

[Installation and launch](../README.md) · [Development and troubleshooting](development.md)

This guide covers dataset preparation, model runs, evaluation, and exports. For a
first run, follow the short workflow in the [README](../README.md#using-the-app).

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

Each dataset folder holds its CSV files, extracted features, split results, and model runs.
Split and model folders are siblings:

```text
ESOL/
  dataset.csv
  GRAPH_graphs/
  FP_fingerprints/
  random_seed_17/
    split_<split identity>.json
    similarity_distribution.csv
    nearest_neighbors.csv
  MODEL_my_model_42/
    run.json
    checkpoint_100.ckpt
    losses.csv
    test_metrics.csv
    evaluation.csv
    predictions.csv
```

Here `17` is the data split seed and `42` is the model seed. Classical models use
`.joblib` files in their model-run folder. Model and split-result folders are created
when results are saved or exported; feature folders are created during extraction.
The filenames above are examples.

The app adds a type prefix to each output directory: `GRAPH_` for extracted 2D or 3D
graphs, `FP_` for fingerprints, and `MODEL_` for model runs. You can enter a short name
such as `graphs` or an already prefixed name such as `GRAPH_graphs`; the matching
prefix is added only once. The model label itself is retained in the run metadata.

Training and prediction graph selectors show only usable `GRAPH_` folders, and
fingerprint selectors show only usable `FP_` folders. Checkpoint selectors search
all `MODEL_` subdirectories for `.ckpt` files in GNN tabs or `.joblib` files in
classical tabs. Root-level checkpoints and folders without the matching prefix are
omitted from the selectors. Existing folders are not renamed automatically.

## Loading datasets

All six workflows provide three checkboxes in one row immediately above **Load dataset**.
They are all checked by default:

| Option | Effect on the imported dataset |
| --- | --- |
| **Canonicalize SMILES** | Write canonical SMILES while preserving stereochemistry and isotopes. |
| **Remove duplicates** | Keep the first row for each molecule, including that row's target values. |
| **Remove salt** | Remove recognized salt fragments from the saved SMILES, retaining the molecule. |

Salt removal runs before duplicate detection. Duplicates are compared by molecular
identity even when canonical SMILES output is disabled; distinct stereoisomers remain
separate. Uncheck **Remove duplicates** to retain replicate measurements or repeated
prediction rows. If both salt removal and canonicalization are unchecked, valid input
SMILES keep their original representation.

Rows with missing required values, invalid SMILES, or molecules excluded by the app's
carbon filter are removed regardless of these options. The loading status reports
the number of molecules imported and removed, with removal reasons. Canonicalizing
a SMILES or stripping a salt fragment does not itself count as removing a molecule.
If no valid molecules remain, the load fails without replacing an existing dataset.

These controls apply to the loaded CSV. Feature extraction and scaffold grouping
retain their existing salt stripping, including when salts were kept at import.
After replacing a dataset or changing its import options, re-extract its features
and process the data again so rows remain aligned. For prediction-only CSVs, leave
the target-column field blank; 3D workflows accept at most one target column.

## Feature extraction

All workflows share the dataset-upload controls. Hybrid and classical tabs share
2D graph and fingerprint extraction controls; 3D tabs use conformer extraction
controls. Choose the loaded dataset and extract the features required by your model:

| Model workflow | Features used for training |
| --- | --- |
| Hybrid GNN | 2D graphs, fingerprints, or both |
| 3D GNN | 3D molecular conformers |
| Classical ML | Fingerprints |

Classical tabs also provide graph extraction for preparing features in the same
working directory; their scikit-learn models continue to use fingerprints only.
For 3D graphs, choose the number of conformers and force field. Use distinct graph
directory names when retaining both 2D and 3D features.

If extraction removes molecules that could not be processed, re-extract the other
features for that dataset before training. The dataset and its feature rows must
refer to the same molecules in the same order. Prediction features must use the
same extraction settings as training features.

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

GNN tabs also provide a graph-cache limit (128 MiB by default), loader workers
(0 by default), and optional pinned memory for CUDA. These take effect when data
are processed. Start with the defaults, then consult [performance and memory](performance.md)
to compare settings. **Release current run**, below the working-directory button,
discards that tab's unsaved results and in-memory resources while preserving its
saved files. Closing or refreshing the browser normally releases its session; save or
export anything you need first. GNN prediction tables preview 200 rows, while CSV
exports contain every prediction.

### Separate data and model seeds

**Data split random seed** is in the split controls of **Load and process data**,
above **Batch size** in the GNN tabs. It controls split selection, GNN training-loader
shuffling, and randomized PCA. Changing it takes effect when data is processed again.

**Model name** and **Model random seed** appear together above **Create model**
(**Fit model** for classical ML). This seed initializes the neural network or seeds
the selected scikit-learn estimator. Deterministic estimators may not use randomness.
Each model captures its name, seed, settings, and processed split when created or fitted.
Changing the controls afterward does not rename an existing run or redirect its files.

Checkpoints, loss data, metrics, evaluation rows, and predictions are saved under
`MODEL_<model name>_<model seed>/`. The run's `run.json` records both seeds, model settings,
dataset and feature identities, and row assignments for later comparisons. Reusing a folder with
different data or model settings is refused; choose another model name to keep the
runs separate. Split-result folders remain `random_seed_<data split seed>/` beside
the `MODEL_` folders.

The saved-model selectors find files inside `MODEL_` folders. Saving selects the new
checkpoint immediately; opening or reopening a working directory rescans its model
folders, including files added outside the app. Checkpoint selections use portable
relative paths on Windows and Linux.

Loading a saved run restores its recorded name and model seed. For GNN models,
create a fresh optimizer after loading a checkpoint before continuing training;
optimizer moments and scheduler state from earlier training are cleared. Training
and evaluation require the original processed split and features. Prediction can
use compatible new data and retains the loaded model's recorded target names.
Hybrid predictions check saved fingerprint type, radius, and bit count when that
metadata is available. Re-extract prediction fingerprints with matching settings
if the app reports missing or incompatible metadata.

GNN loss plots and exports use completed epoch numbers starting at 1. After loading
a checkpoint, they contain only the subsequent training history, with epoch numbers
continuing from the saved checkpoint.

The block rename also changes hybrid neural-network checkpoint parameter keys to
`graph_encoder`, `fingerprint_encoder`, and `prediction_head`. Old parameter keys
are not migrated; create and save new models for the renamed architecture.

### Training–test similarity

At the bottom of **Load and process data**, process a split and click
**Calculate training–test similarity**. The calculation uses Morgan fingerprints
from the captured molecular SMILES, with configurable radius and bit count, and
Tanimoto similarity. It is available in all six tabs, including graph-only workflows.
It uses the extractor's salt removal and hydrogen handling and ignores chirality;
model feature selection and PCA do not affect this comparison.

The plots show the distribution of all training–test pair similarities and each
test molecule's nearest training neighbor similarity. Validation rows are excluded.
The neighbor table identifies the matched molecules and their zero-based dataset
row numbers. Ties select the lowest training row number.

Choose filenames and export the distribution counts or nearest-neighbor rows to
`random_seed_<data split seed>/`. Each CSV has a companion `.csv.metadata.json`
recording the split, fingerprint settings, and summary. The folder uses the seed
captured during processing, even if the slider has since changed. Processing another
split clears the previous similarity results; creating or training a model retains them.
The calculation compares every training–test pair in bounded chunks, so large datasets
can take time without requiring a full pairwise matrix in memory.

### Evaluation metrics

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
button. Files are saved in the model's captured `MODEL_<model name>_<model seed>/` folder
inside its working directory. Names may include
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
and extract fingerprints. Existing aligned `FP_` fingerprint folders can also be used.
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

