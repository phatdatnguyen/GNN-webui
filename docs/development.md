# Development and troubleshooting

[README](../README.md) · [User guide](user-guide.md)

Use the project's Python 3.11 environment. The repository's
[AGENTS.md](../AGENTS.md) describes handler contracts, session state, data alignment,
checkpoint metadata, and testing conventions in detail.

## Shared dataset preparation

`update_dataset.py` and `feature_extraction.py` provide the common dataset-upload
and extraction controls and handlers used by the six workflows. Keep shared UI
changes there; workflow modules retain their model-specific processing and state
reset wiring. Classical model fitting still consumes fingerprints even when its
feature-extraction section is used to prepare graphs. The RDKit/DeepChem extraction
and CSV import implementation remains shared in `utils.py`.

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
pytest tests/test_shared_data_sections.py tests/test_dataset_upload.py  # shared upload/extraction across six tabs
pytest tests/test_classical_api.py tests/test_evaluation_exports_api.py  # all six export flows through Gradio
pytest tests/test_checkpoint_refresh_api.py tests/test_prediction_targets.py  # checkpoint refresh and prediction labels
pytest tests/test_loss_epochs.py tests/test_prediction_features.py  # resumed epochs and fingerprint compatibility
pytest tests/test_run_utils.py tests/test_similarity.py  # run provenance and similarity calculations
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
The underlying native crash remains undiagnosed, and this setting does not eliminate
every occurrence; no dependency versions were changed to work around it. Keep the
crash log when investigating the environment.

The MTX CUDA investigation also reproduced a training crash with this workaround
enabled, and reproduced a Python-only crash without the app or GPU libraries.
See [the investigation and standalone Python 3.11 check](../diagnostics/segfault-investigation.md)
before treating an allocator or architecture change as a fix.
