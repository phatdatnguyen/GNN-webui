# GNN performance and memory

[User guide](user-guide.md) · [Development](development.md)

The four GNN workflows retain their extracted graph files and fingerprint CSVs.
Processing prepares numeric data in memory, and a bounded CPU graph cache avoids
deserializing the same molecules repeatedly. Existing row-alignment, feature-content,
split, and checkpoint checks still apply. Python 3.11 remains the supported runtime.

## Data preparation and cache settings

Fingerprints and targets are converted to contiguous numeric arrays once. Binary
fingerprints use one byte per bit value rather than an eight-byte integer; this is
about 20 MiB for 10,000 molecules with 2,048 bits, excluding SMILES, prepared features,
and temporary batches. Legacy nonbinary fingerprint values retain integer storage.
Regression targets retain their raw precision until scaling.

Variance filtering, PCA, and target scaling are fitted on training rows only.
Transformed data are prepared once per fitted transformer state and reused by
subsequent batches. Loading a checkpoint with different transformers causes those
prepared values to be rebuilt. Predictions use the model's fitted transformers.

The graph-cache budget defaults to **128 MiB per processed dataset**. Training,
validation, and test subsets share that cache. Prediction data have their own budget.
The budget covers estimated cached graph objects and tensor storage, not the application's
total RAM: numeric arrays, models, optimizer tensors, batches, plots, and Python
imports consume additional memory. Set the budget to **0** for disk loading. Cached
graphs are copied before returning them, so attaching targets or modifying a sample
cannot change the source retained for another batch.

The default is **0 loader workers** and **unpinned memory**. After checking those
defaults, compare 2 or 4 workers and pinned memory on your own hardware. Worker
processes use `spawn`, a bounded prefetch queue, and no persistent workers. They
do not maintain additional graph caches. A fully cached dataset can be faster with
zero workers, while disk loading may benefit from workers. Windows/WSL startup and
copies of dataset arrays can make multiprocessing expensive. PyTorch documents
these worker-memory and pinning tradeoffs in its
[DataLoader guide](https://docs.pytorch.org/docs/stable/data.html).

The **Graph cache limit (MiB)**, **DataLoader workers**, and **Pin memory for CUDA
transfers** controls appear in **Load and process data**. Settings take effect at
**Process data** or **Process prediction data**. Keep batch size, split seed, model
seed, architecture, and device unchanged when comparing timings. The first graph
pass fills an application cache; subsequent passes can be faster. The operating
system also caches files, so a fresh application cache does not imply a cold disk cache.

## Releasing resources

Creating a new model clears the previous optimizer and scheduler as well as its
results. Both objects must be cleared: the scheduler refers to the optimizer, and
the optimizer refers to old parameters, gradients, and optimizer tensors. Loading a
checkpoint continues to require creating a fresh optimizer before training.

Use **Release current run**, below **Create/Open Working Directory** in a GNN tab,
when finished with that workflow. It discards processed data, models, training
state, similarity, and results from memory. Saved files and the selected working
directory remain available. Reprocess the data before starting another run. This
is also useful before constructing a substantially larger model, since normal
replacement can briefly hold the old and new models simultaneously.

Prediction previews show the first **200 rows**. All prediction rows stay on the
server for complete CSV export; the preview limit does not change predictions or
evaluation metrics. Releasing a run discards unsaved results.

When Gradio detects a session disconnect, the app detaches the session resources,
including rendered tables and plots, and removes queued jobs for that session.
Disconnect detection can be delayed after a network interruption. Reopening starts
a new in-memory session; save checkpoints
and export results before leaving. A running callback keeps the references it needs
until it finishes, so cleanup does not invalidate tensors during training. Saved
files remain available. If an installed Gradio version is incompatible with the
cleanup adapter, the app warns and falls back to Gradio's normal state expiry.
Use **Release current run** for an explicit release while the page is connected.

A high RAM reading does not by itself demonstrate a leak. Python/native allocators
may retain reusable pages after objects are released. On CUDA, distinguish live
tensor memory (`memory_allocated`) from allocator reservations (`memory_reserved`).
`empty_cache()` releases unused reservations; it cannot release tensors still held
by an optimizer or another object. See
[PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).
Garbage collection and CUDA cache clearing do not run inside training batches.

## Reproducible diagnostics

Run the diagnostic with the existing Python 3.11 environment. With no input directory,
it creates disposable synthetic feature files and removes them afterward:

```bash
./gnn-env/bin/python diagnostics/benchmark_gnn.py --rows 512 --cache-mb 0 128
```

To compare an existing dataset without changing any source files:

```bash
./gnn-env/bin/python diagnostics/benchmark_gnn.py \
  --workdir Data/Classification/MTX \
  --graphs GRAPH_MGCF --fingerprints FP_AvalonFP_2048 \
  --rows 512 --batch-size 64 --cache-mb 0 128 --workers 0 \
  --output /tmp/gnn-benchmark.json
```

Use `--workers 0 2 4` to compare loading configurations, `--device cuda --pin-memory`
to test pinned CUDA transfers, or `--mode 3d` for a SchNet/conformer diagnostic.
`--mode graphs` and `--mode fingerprints` isolate the corresponding inputs.
For regression, add `--task regression`; `--pca-components 32` enables a small PCA
comparison when fingerprints and enough sampled rows are available. These are small
representative training models, not a reproduction of every UI architecture.

Each configuration reports:

- Dataset construction and preprocessing times, followed by separate first and
  repeated iterator passes over the same seeded sample of rows.
- Complete training-epoch time, rows per second, and host time spent waiting for
  batches, transferring inputs, and running forward/backward/update operations.
  CUDA epoch timing synchronizes at its boundaries; CUDA host-stage times describe
  submission/wait overhead rather than GPU kernel durations.
- RAM RSS/USS, CPU time, thread/child-process counts, and file descriptors when the
  optional `psutil` package is installed; CUDA allocated/reserved/peak memory when
  using CUDA. Memory snapshots are observations, not a complete CPU peak profiler.
- Whether weak references to the dataset, loader, model, optimizer, scheduler,
  parameter, gradient, and Adam moment disappear after the configuration is released.

Use `--cycles 20` with a small sample to look for continuing resource growth after
warm-up. Counts and memory should stabilize; immediate return to the process's
initial RSS or reserved CUDA memory is not required. This CLI verifies standalone
ownership. The Gradio API tests additionally cover model replacement, workflow
release, full exports, and isolation between browser sessions.
With workers enabled, Python may keep one multiprocessing resource tracker and
IPC infrastructure until process exit. The report identifies child roles so that
this infrastructure is not mistaken for a training worker that failed to stop.

Results are printed as JSON. `--output` additionally writes a new report and refuses
to overwrite an existing file. The diagnostic never saves trained models or edits
input data. The default native thread limits are conservative; existing
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` values are respected.
`--threads` controls PyTorch's CPU thread count.

Compare complete epochs before claiming a training speedup. A faster row-indexing
microbenchmark does not establish an equal improvement in GPU-bound training. The
existing [native-runtime investigation](../diagnostics/segfault-investigation.md)
remains separate from data-loading performance.

## Example measurements

A CPU-only check on the existing MTX data used 512 seeded rows from 5,804 molecules,
2,048-bit Avalon fingerprints, batch size 64, one CPU thread, and zero loader workers.
The same row sample was iterated three times:

| Data path | First iterator pass | Third iterator pass |
| --- | ---: | ---: |
| Previous pandas conversion and per-sample graph loading | 0.985 s | 0.936 s |
| Prepared arrays, graph cache disabled | 0.189 s | 0.085 s |
| Prepared arrays, 128 MiB graph cache | 0.137 s | 0.034 s |

These timings cover iteration, not GPU training. In the updated code, a small GCN's
second training epoch over those rows took 0.110 s without graph caching and 0.059 s
with caching. This is a comparison of two new-code configurations, not an old versus
new training benchmark. The application cache held the 512 sampled graphs.
These initial CPU measurements ran in a sandbox that hid CUDA from the process.

Twenty create/train/release cycles on a small synthetic GCN released all tracked
dataset, loader, model, optimizer, scheduler, parameter, gradient, and Adam-moment
references. RSS stabilized at about 954.5 MB after the second cycle, with one thread,
no child processes, and four file descriptors. A separate 20-cycle fingerprint/PCA
regression check also released all tracked references. These are standalone CPU
ownership checks; large production models and browser sessions require their own
measurements.

A subsequent run outside that sandbox used an NVIDIA GeForce RTX 4090 with
PyTorch 2.12.0+cu132. MTX CUDA training completed with graph caching enabled/disabled
and pinned memory enabled/disabled; the two-epoch losses agreed within `1e-6` across
configurations. CUDA epoch times varied substantially, including first-use startup,
so these short runs do not establish a reliable GPU speedup or a preferred pinning
default. Ten small GCN create/train/release cycles released every tracked reference
and kept post-release CUDA allocation at 16.25 MiB and reservation at 22 MiB. Stable
runtime allocations can remain after model objects are gone; the diagnostic checks
growth after warm-up rather than requiring zero GPU allocation.
