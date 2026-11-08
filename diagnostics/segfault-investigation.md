# MTX CUDA crash investigation — 2026-11-08

Python 3.11 remains the project's environment. The temporary Python 3.12 migration
environment was removed when that investigation was cancelled. No model, dataset,
installed package in `gnn-env`, or training setting was changed by this investigation.

## Findings

- All 5,804 MTX CSV rows, graph records, graph-manifest entries, and fingerprint rows
  agree on SMILES and ordering. All graphs have 33 node features and 11 edge features;
  features are finite float32 and edge indices are valid int64. The 2,048 Avalon
  fingerprint columns contain finite binary values. Targets are 4,734 negatives and
  1,070 positives.
- The default hybrid model was reproduced on the RTX 4090: GCN hidden width 128,
  three layers, 50 outputs; fingerprint encoder width 128, three layers, 50 outputs;
  prediction head width 128, one layer; float32, batch size 32, seeds 0, 20% validation
  and test, Adam at 0.0001 with 0.99 learning-rate decay. No outputs were saved.
- With a clean library environment, `PYTHONMALLOC=malloc`, and CPU affinity 16,
  nine epochs completed and training segfaulted during epoch 10. The traceback was:

  ```text
  pandas/core/internals/managers.py: fast_xs
  pandas/core/frame.py: _ixs
  pandas/core/indexing.py: __getitem__
  datasets.py: MolecularDataset.__getitem__
  torch DataLoader
  binary_classification_hybrid.py: train
  ```

  This is fingerprint row loading (`fingerprint_df.iloc[idx]`), before that batch
  reaches the GPU. The user's actual `python webui.py` core has the same Python stack;
  its native fault is inside CPython's `_PyEval_EvalFrameDefault`, dereferencing a
  null interpreter-stack value. This identifies the failure site, not its root cause.
- Other attempts segfaulted during Gradio/Jinja2 imports before loading the dataset.
  An earlier user core failed during SciPy docstring parsing on import.
- An isolated Python 3.11 process using only Jinja2/MarkupSafe `.py` sources also
  segfaulted. It used `-I -S` with no application, pandas, PyTorch, CUDA, or third-party
  native extensions. Clearing library overrides and using CPU affinity 16 did not
  reliably prevent that failure.
- Torch and compiled PyG extension version tags agree, and `pip check` found no
  dependency conflicts. GPU availability succeeds outside the execution sandbox.
  The shell's `LD_LIBRARY_PATH` selects ORCA's C++ runtime; clearing it selects the
  system runtime, but the crash also reproduces with those overrides removed.

These results demonstrate instability outside the model and dataset path. They do
not establish whether the remaining cause is the interpreter build, WSL/host software,
CPU/RAM instability, or another system-level problem. They do not justify changing
the architecture, learning rate, or random seed as a fix. Caching numeric features
could remove the slow pandas row-conversion path, but would not fix the independently
reproduced Python-only failures.

## Repeat the small check

```bash
./gnn-env/bin/python diagnostics/check_python311_runtime.py --seconds 20
```

The script copies only Python package sources to a temporary directory, runs an
isolated child with fault reporting enabled, and cleans up afterward. A crash without
the ML stack is useful evidence when checking system changes. One successful run is
not proof that the underlying intermittent problem is resolved.

To capture the app's next fault without changing training settings:

```bash
PYTHONFAULTHANDLER=1 ./gnn-env/bin/python -u webui.py 2>&1 | tee /tmp/gnn-webui-crash.log
```

The allocator workaround previously documented in README is not a reliable cure:
the epoch-10 failure occurred with it enabled.

## Next checks outside the application

The host reports an Intel Core i9-14900K and a Gigabyte Z790M AORUS ELITE AX motherboard
with BIOS FL. Verify the motherboard revision and current firmware/settings with the
manufacturer. Intel documents instability affecting some 13th/14th-generation desktop
processors and recommends the appropriate Intel Default Settings and current BIOS
containing microcode 0x12F or later. This is a reason to investigate host stability,
not evidence that this particular CPU has that defect.
[Intel guidance](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html).

Gigabyte's revision 1.2 support page lists a newer FM BIOS than the reported FL and
lists 0x12F support in an earlier FI release. Therefore, do not assume that FL lacks
the original mitigation, and select firmware only for the verified board revision.
[Gigabyte support](https://www.gigabyte.com/Motherboard/Z790M-AORUS-ELITE-AX-rev-12/support).

Check Windows hardware-error records and CPU/RAM stability with the motherboard or
system vendor. Comparing the small Python 3.11 reproducer under native Windows and
WSL can help distinguish a WSL-specific problem from a host-wide one. No firmware,
voltage, clock, or system settings were changed during this investigation.
