# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

The virtualenv lives at `gnn-env/` (Python 3.11). Everything below assumes it is activated,
or prefix with `./gnn-env/bin/python -m`.

```bash
python webui.py                      # start the app (picks the first free port from 7860)

pytest                               # full suite
pytest -m "not slow"                 # skip tests requiring 3D conformer generation
pytest tests/test_models.py          # one file
pytest tests/test_handlers.py::test_checkpoint_round_trip_restores_weights   # one test
pytest -k checkpoint                 # one topic across all files
```

There is no linter or formatter configured, and no build step.

`torch` is installed separately from `requirements.txt` (it must match the local CUDA
version — see README). `deepchem>=2.8.0` is a hard floor: `PagtnMolGraphFeaturizer` and
`DMPNNFeaturizer` do not exist in earlier releases and the featurizer dropdown offers both.

## Architecture

### Tab modules

`webui.py` builds one Gradio `Blocks` with six tabs and mounts it on FastAPI. Each tab is
a self-contained module:

| Module | Task | Inputs |
|---|---|---|
| `binary_classification_hybrid.py` | BCEWithLogits, multi-target | 2D graphs and/or fingerprints |
| `regression_hybrid.py` | MSE, multi-target, MinMaxScaler on targets | 2D graphs and/or fingerprints |
| `binary_classification_3d.py` | BCEWithLogits, single target | 3D conformers |
| `regression_3d.py` | MSE, single target, MinMaxScaler | 3D conformers |
| `binary_classification_classical.py` | Classical binary classification | Molecular fingerprints |
| `regression_classical.py` | Classical regression | Molecular fingerprints |

**The four GNN modules are near-duplicates by design.**
A change to a handler almost always needs mirroring in the other three. The pairs differ in:
the loss function, the metrics block, whether an `output_scaler` is threaded through, and
2D (graph + fingerprint branches) versus 3D (SchNet/DimeNet++/ViSNet, positions only).

Dataset upload and feature extraction are exceptions: their controls and common
handlers live in `update_dataset.py` and `feature_extraction.py`. All six tabs use
`build_dataset_upload_controls`; 3D tabs pass `is_3d=True` and retain their single-target
restriction. `build_feature_extraction_controls` builds 2D graph/fingerprint controls
for hybrid and classical tabs, or the conformer/force-field controls for 3D tabs.
The returned dataclasses contain the components needed for workflow event wiring.
Keep widget defaults, import validation, extraction error handling, and fingerprint
radius visibility in these shared modules. Thin workflow adapters retain the existing
callback signatures and positional return contracts. The chemistry and atomic file
publication routines remain centralized in `utils.py`.

Shared code lives in `utils.py`, which each GNN tab pulls in with `from utils import *` —
that is how `DATA_DIR`, `sanitize_path_component` and `list_graph_files` become available
as module globals in each tab (relevant when monkeypatching in tests).

The classical tab modules are thin entry points sharing `classical_ui.py` (Gradio
handlers and session state), `classical_ml.py` (aligned data, fitting, evaluation,
and checkpoints), and `classical_catalog.py` (scikit-learn models, parameter controls,
and model constraints). Add models in the catalogue so both the dropdown and its
parameter controls update together. Each target gets its own fitted estimator.
Feature scaling and any probability calibration fit only training rows; saved
models retain preprocessing, fingerprint settings, target names, and a hash of the
data and split assignments. Evaluation rejects a model from a different split;
prediction accepts compatible new data.

Classical workflows split into training and test sets only (20% test by default).
Their shared feature panel also offers 2D graph extraction (float32), while model
fitting and prediction continue to consume fingerprints only. Extracting graphs from
the active training CSV invalidates its processed data/model/results/similarity,
because extraction can remove CSV rows. Unrelated prediction extraction preserves
the training run. Matching extraction failures also invalidate state conservatively
in case publication changed source files.
Their evaluation caches per-row actual values and predictions for output selection
and plots. All evaluation plots share the right column of **Model evaluation**;
all six tabs use **Make predictions** for prediction controls and results.

All six tabs provide separate metrics and evaluation-row CSV exports. Metrics use
the shared `evaluation_metrics.metrics_from_predictions` helper and the schema
`Target,Metric,Value`; row exports contain SMILES, actual values, predictions, and
classification probabilities. Exports include all targets regardless of the displayed
output and reuse cached evaluation rows. `utils.export_result_csv` validates nonempty
results, the working directory, and the filename, and accepts names with or without
`.csv`. `classical_plots.py` renders classical evaluation plots from those cached rows.

### The handler contract

Every `on_*` function is wired to a Gradio component event at the bottom of
`<module>_tab_content()`. Two rules govern them:

1. **Handlers report errors, they do not raise.** Each returns an HTML status string
   coloured green on success or red on failure, wrapped in `try/except`. Raising instead
   surfaces a raw traceback in the browser. Tests assert on this via `is_ok` / `is_error`
   in `tests/conftest.py`.
2. **The return tuple must line up positionally with the `outputs=[...]` list** in the
   corresponding `.click()` / `.change()` call. Adding a return value means editing the
   wiring. `tests/test_webui.py` checks the Blocks tree and component wiring; runtime
   return counts and postprocessing require the Gradio API tests in
   `tests/test_classical_api.py` and `tests/test_evaluation_exports_api.py`.

Error paths must return `gr.update(...)` for component slots (not `None`, which blanks the
component) and must echo back unchanged state for `gr.State` slots — several handlers take
the current losses/epoch count as inputs purely so they can return them untouched on failure.

### Session state

All mutable state is per-browser-session `gr.State`: dataset, the three dataloaders, model,
optimizer, scheduler, loss history, epoch count, and the fitted transformers
(`VarianceThreshold`, `PCA`, and `MinMaxScaler` for regression). Handlers are otherwise
pure — they take state in and hand new state back. There are no module-level mutables
apart from `loss_fn`.

`ui_utils.bind_data_processing` clears dependent GNN model, training, prediction, and
evaluation state atomically when processing a split. `bind_result_invalidation`
clears evaluation/prediction results and disables their exports after model creation,
checkpoint loading, or training. It also clears on errors because a failed action
can discard the old model or update some weights before reporting failure. These
wrappers append UI outputs while preserving direct handler APIs; avoid assuming
the original model/history states are the final outputs of the wrapped callback.

Split and model seeds are independent. `run_utils.make_split_context` captures the
processed dataset hash, SMILES, row assignments, method, and data seed. GNN datasets
carry `split_context`; classical `ProcessedData` does too. `make_run_context` captures
the model name, model seed, workflow, parameters, and split. GNN models use
`_run_context`; classical `FittedModel` uses `run_context`. Export paths must use
these snapshots, never live name/seed controls. Data seeds also seed randomized PCA
and GNN loader generators; model seeds initialize models/estimators.

`gnn_run_ui` wraps GNN processing, model actions, exports, and working-directory
changes while preserving direct handler APIs. Its extra outputs include split state,
similarity resets, and the read-only run-directory display. The shared concurrency ID
serializes actions affecting a tab's run. Processing or changing working directories
clears similarity as well as models/results; model creation/loading/training retains
similarity. `similarity_ui` builds and binds the common controls for all six tabs.

Opening a directory and saving a checkpoint refresh checkpoint selectors in the same
callback response. Do not rely solely on the root contents `gr.State.change`: a new
file inside an existing `MODEL_` folder leaves that state's value unchanged. Successful
saves select their new file; general refreshes preserve the selected checkpoint.
Checkpoint dropdown identifiers always use `/`, including on Windows, to match the
safe path resolver. `tests/test_checkpoint_refresh_api.py` exercises repeated saves,
reopening, and switching directories with identical root names through Gradio.

Two pieces of state are bound to a specific model object and go stale if it is replaced:

- **The optimizer captures `model.parameters()` when it is created.** Train with an
  optimizer built for a previous model and `backward()` fills gradients on the new model
  while `step()` walks the old parameter list, so nothing updates and the loss sits
  perfectly flat while the status still reports success. `on_create_model` therefore
  returns `interactive=False` for `train_button`, forcing a fresh optimizer, and
  `utils.ensure_optimizer_matches_model` re-checks at the top of `on_train`.
  The UI also clears both optimizer and scheduler on every model creation attempt;
  the scheduler itself keeps the old optimizer and its tensors alive otherwise.
- **The device is applied once, at creation**, but the Device dropdown is read on every
  run. `utils.ensure_model_on_device` turns the resulting tensor-device mismatch into a
  readable error instead of a raw `RuntimeError`.

`on_train` appends the train and validation loss together and bumps `trained_epochs` only
after a full epoch, so a failure part-way through leaves the loss lists and the counter in
agreement. It catches its own exceptions and still returns a plot of the completed epochs.

All four GNN tabs expose **Release current run**, which clears processed data/loaders,
models, transformers, histories, similarity and results under the tab's concurrency
lock while preserving its working directory and saved files. Prediction results live
in a separate `gr.State`; the displayed table is a 200-row preview and CSV export uses
the full state. Include this state in every relevant invalidation path.
`session_lifecycle.register_session_cleanup` binds Gradio's unload event on the root
Blocks, including when mounted on FastAPI. Its guarded adapter detaches the complete
session (including rendered payloads); active callbacks keep their own references until
they finish. It must not mutate in-flight tensors or another session's resources.

### On-disk layout and the row-alignment invariant

`utils.process_dataset_file` handles dataset import for all six tabs. Its optional
`canonicalize_smiles`, `remove_duplicates`, and `remove_salt` arguments default to
`True`; the existing sixth positional `is_3d_dataset` argument is retained. The
checkboxes come from `update_dataset.dataset_cleanup_controls` in one row above
**Load dataset** (`ui_utils` retains a compatibility import).
Salt removal precedes deduplication; canonical isomeric molecule identity determines
duplicates even when canonical output is disabled. Keep the first row and its target
values. Status counts partition the source rows into imported and removed molecules;
salt fragments and rewritten SMILES are not removed rows. All-invalid imports preserve
the existing dataset. Extraction/scaffold salt handling is independent of these
import-only options. Tests in `tests/test_dataset_upload.py` exercise all six upload
callbacks through Gradio, including checkbox defaults and error paths.

```
Data/{Classification,Regression}/<working_dir>/
  <dataset>.csv                      SMILES column + target column(s)
  GRAPH_<graph_dir>/<i>.pt           one torch-saved PyG Data per row
  GRAPH_<graph_dir>/smiles.txt       manifest: the SMILES each graph was built from, in order
  FP_<fp_dir>/molecular_fingerprints.csv   SMILES + Bit_1..Bit_N, one row per row
  random_seed_<data_seed>/
    split_<split_id>.json
    <similarity_export>.csv
    <similarity_export>.csv.metadata.json
  MODEL_<model_name>_<model_seed>/
    run.json
    <checkpoint>_<epochs>.ckpt       GNN model
    <saved_model>.joblib            classical model
    <losses_or_metrics_or_evaluation_or_predictions>.csv
```

Extraction adds `GRAPH_` or `FP_` to the requested feature directory name unless that
exact prefix is already present. Training and prediction dropdowns list only usable
directories with the matching prefix. Model folders similarly use `MODEL_` exactly
once; the captured human model label remains unchanged. A label that already starts
with `MODEL_` and one without it can resolve to the same directory, so the manifest
collision check must still reject different run identities sharing that destination.
Existing folders are not migrated or renamed.

Split and model-run directories are siblings, as requested by the user. `run_utils`
validates path components and manifests and discovers checkpoints only inside immediate
`MODEL_` subdirectories; root-level and other unprefixed files are not dropdown choices.
Safe root-level checkpoint paths remain supported by direct handler callers. The helpers
reject symlink/traversal paths. Run/split IDs exclude the absolute
working-directory root so copied runs remain portable. Saved manifests omit that root;
loading explicitly rebases it to the selected working directory. A run folder cannot
be reused for different split/configuration/provenance metadata. The `MODEL_` prefix
keeps model names such as `random_seed` separate from the data-split folders. Save/export
creates split and model-run directories; creating/fitting a model alone does not write them.

GNN processing also captures feature-content identities. Fingerprint contents/settings
and graph shards are hashed once during processing, then recorded with the model run.
Training/evaluation checks these captured feature identities as well as the split;
re-extracting different features into the same directory must not silently reuse an old
run or checkpoint. Replacing an active dataset or feature source through the UI clears
the dependent split, similarity, model, and result state. Unrelated prediction uploads
preserve the training run.

`similarity.calculate_similarity` uses captured SMILES to compute exact Morgan/Tanimoto
training–test comparisons independently of the model's inputs or PCA. Its salt stripping,
explicit hydrogens, and ignored chirality match fingerprint extraction. Validation rows
are excluded. Pairwise histograms are accumulated in bounded chunks; the full pair matrix
is never retained. Nearest neighbors follow captured test order, with ties resolved by
the smallest original training row index. Exports contain distribution bins or per-test
neighbors plus JSON provenance and use the captured split seed. Reprocessing invalidates
the result; changing fingerprint controls does not relabel an already calculated result.

**Row `i` of the CSV, graph `i.pt`, and fingerprint row `i` must describe the same
molecule.** Both extractors rewrite the dataset CSV when a molecule fails, so running one
after the other can silently shift the other artifact. `datasets.py` enforces the invariant
up front by comparing counts *and* SMILES; a mismatch raises telling the user to re-extract.
The graph check reads `smiles.txt` so the whole ordering is verified from one small file;
directories extracted before the manifest existed fall back to spot-checking the first,
middle and last graph. `__len__` counts only files named `<int>.pt` and caches the result,
because a `DataLoader` calls it every epoch.

`datasets.py` prepares contiguous fingerprints and targets once. Validated binary
fingerprints use uint8 storage; legacy nonbinary data retain integer conversion
semantics. The processed numerical data are snapshots: reprocess after changing their
source. Train/validation/test Subsets share a bounded CPU graph cache (128 MiB by
default, 0 disables it). Cached graphs are copied before returning samples, and the
budget accounts for tensor storage and Python objects, not total application RAM.
Spawned workers do not receive or maintain copies of this graph cache.

`prepare_dataset` prepares PCA/variance-filtered fingerprints and scaled targets in
chunks after fitting on training rows only. It hashes actual fitted transformer state
and dtype to rebuild views when checkpoints change preprocessing. `.fp` and `.y`
retain their original meaning; `.fp_prepared` and `.y_prepared` are optional transformed
views. Preserve raw regression precision before scaling. The model-summary path leaves
target preparation intact. Generic PyG datasets retain batch-time transform fallbacks.

`gnn_runtime.make_loader` centralizes seeded loaders, optional pinned memory and
0/2/4 workers. Defaults remain 0 workers and unpinned memory. Multiprocessing uses
spawn, bounded prefetch and nonpersistent workers; validate shutdown on error as well
as completion. Training accumulates detached weighted loss on-device and reads it once
per epoch, while retaining the finite-loss check before backward. Do not introduce
per-batch garbage collection or CUDA cache clearing.

The training loader shuffles each epoch, seeded from the data split random seed so runs stay
reproducible; the "Shuffle training data each epoch" checkbox turns it off for a fixed
batch order. Validation and test loaders are always in dataset order.

Names typed into the working-directory, dataset, checkpoint, graph-directory and export
textboxes all reach `os.path.join`, and the extraction directory is passed to
`shutil.rmtree`. Everything user-supplied goes through `utils.sanitize_path_component`,
which rejects unsafe path components rather than rewriting them. The type-prefix
helpers then add the required folder prefix exactly once.

### models.py

The hybrid models have the same shape: **Graph Encoder + Fingerprint Encoder → concat →
Prediction Head → output layer.** Either encoder is disabled by setting its `n_outputs`
to 0 (both zero raises).
Fingerprint-only datasets use `MLPModel` instead of a graph model.

The registered PyTorch block attributes are `graph_encoder`, `fingerprint_encoder`, and
`prediction_head`, so these names appear in `repr(model)`, `named_modules()`, and saved
parameter keys. Algorithm class names (`GCNModel`, `MLPModel`, etc.) and constructor
argument names (`n_gcn_*`, `n_mlp_*`, `n_predictor_*`) are retained. Keep all forward
references synchronized with the registered attributes; do not register aliases under
the old `gcn`, `mlp`, or `predictor` names. The rename does not alter layer order or
tensor flow. PyG 3D models and scikit-learn estimators retain their algorithm names.

Two families:

- **Top-level** (`GCNModel`, `GraphSAGEModel`, `GINModel`, `GATModel`, `EdgeCNNModel`,
  `AttentiveFPModel`) wrap PyG's `BasicGNN` classes. Watch the positional signature:
  `BasicGNN.forward(x, edge_index, edge_weight, edge_attr, batch, ...)` — it is easy to
  land an argument in the wrong slot.
- **Custom** (21 conv layers behind the "Custom" tab) build `ModuleList`s by hand. The
  simple ones alternate conv/`Linear`; the edge-aware ones alternate conv/`Linear` in pairs
  and `TransformerConv` uses conv/`Linear`/`BatchNorm1d` triples. **`forward()` decodes the
  list by index parity** (`i % 2`, `i % 3`), so the construction order and the forward loop
  are tightly coupled — changing one without the other silently mis-applies layers. The
  last `Linear` in each stack must project to `n_gcn_outputs`, including when
  `n_layers == 1` and the build loop never runs.

`models.model_requires_edge_attr(model)` decides whether `forward` receives an
`edge_attr` argument from the actual model class. Creation, training, validation,
evaluation, and prediction use this helper, so changing the UI model selection cannot
change how an existing model is called. Keep it in sync with models that accept edge
features. A PyG layer given `edge_attr` but constructed without `edge_dim` can discard
it silently rather than erroring.

### Dtypes

The Data-type dropdown (float16/32/64/bfloat16) applies to **node/edge features and
targets only**. `edge_index` and the 3D atomic numbers `z` are indices and are always
stored as `torch.long` — storing them as float16 corrupts values above 2048 with no error.
`datasets.py` casts on load so graphs extracted by older versions still work.

### Checkpoints

`.ckpt` files hold `model_class`, `state_dict`, `trained_epochs`, and the fitted
transformers. The transformers are part of the model's contract — a regression checkpoint
reloaded without its `output_scaler` denormalises predictions with the wrong constants and
produces plausible but wrong numbers. Architecture is checked on load via `model_class`;
loading into a mismatched model is refused. Older checkpoints missing these keys fall back
to the current session state. Named checkpoints also retain `run_context`.
Hybrid checkpoints using the old `gcn.*`, `mlp.*`, or `predictor.*` parameter keys are
not migrated to the renamed encoder/head attributes; they fail the normal key check.
`utils.load_model_checkpoint` accepts an optional metadata validator that runs before
copying weights. `gnn_run_ui.load_run_checkpoint` verifies embedded and sibling run
metadata, restores the recorded identity, and guards subsequent training/evaluation
against a different split. Legacy GNN imports into a named run record
`initialization_seed_verified=False`; do not claim that their original seed is known.
Legacy classical imports use their saved estimator seed and require a matching processed
training identity to adopt a named run for save/export. Predictions remain available
without adopting a run.

Successful GNN checkpoint loads discard optimizer/scheduler state and disable Train
until a fresh optimizer is created, clear the old loss plot, and disable loss export.
Failed loads preserve those objects. Loss plots and exports use 1-based cumulative
epoch numbers, including when the current history begins after a loaded checkpoint.
`on_export_losses` accepts an optional completed epoch count; the UI supplies it,
while direct callers without it number the provided history from 1. Prediction
column labels come from captured model target names when present, with the existing
dataset fallback for legacy models. Invalid target metadata or output shapes must
report an error rather than silently relabel predictions.

Hybrid prediction processing captures fingerprint configuration separately from
training content identities. Before inference, compare its input mode and available
fingerprint settings with the loaded run; do not compare row/content hashes, because
prediction accepts new molecules. Known saved settings require matching prediction
metadata; legacy models without recorded settings retain input-shape checks.

## Tests

`tests/conftest.py` does the expensive work once per session (feature extraction, conformer
generation) and copies a fresh directory per test. It also provides `create_model_args()`
and `create_model_3d_args()`, which build the 40-odd positional argument lists that
`on_create_model` takes — use these rather than hand-writing positional calls.

Tests needing 3D conformers are marked `@pytest.mark.slow`.

`tests/test_shared_data_sections.py` exercises the common preparation controls and
classical graph extraction through Gradio, including refresh and source invalidation.
User-facing details are in `docs/user-guide.md`; test commands and runtime diagnostics
are in `docs/development.md`. Keep the README below startup focused on the basic workflow.
`docs/performance.md` describes cache/resource controls and the read-only
`diagnostics/benchmark_gnn.py` benchmark, including repeated release checks.

`tests/test_models.py` is deliberately combinatorial: all 21 custom layers × `n_layers`
∈ {1, 2, 3, 6}, because a wrong output width only surfaces at forward time. Several tests
are pinned to specific past bugs and say so in a comment — keep those comments when editing.
