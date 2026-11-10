"""Read-only GNN data/training benchmark; run with the project's Python 3.11.

Without --workdir this creates disposable synthetic features in a temporary folder.
JSON reports distinguish cold application caches from an uncontrollable OS file cache.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import weakref

# Keep BLAS pools conservative in this diagnostic and spawned loader processes.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def snapshot(device):
    import torch

    result = {}
    try:
        import psutil
        process = psutil.Process()
        children = process.children(recursive=True)
        child_roles = []
        for child in children:
            try:
                command = " ".join(child.cmdline())
                child_roles.append("multiprocessing resource tracker" if "resource_tracker" in command
                                   else "worker or other child")
            except psutil.Error:
                child_roles.append("exited child")
        result.update(rss_bytes=process.memory_info().rss,
                      cpu_seconds=sum(process.cpu_times()[:2]),
                      threads=process.num_threads(),
                      children=len(children), child_roles=child_roles)
        try:
            result["uss_bytes"] = process.memory_full_info().uss
        except (AttributeError, psutil.Error):
            pass
        if hasattr(process, "num_fds"):
            result["file_descriptors"] = process.num_fds()
    except ImportError:
        result["process_memory"] = "Install optional psutil for RAM/process measurements."
    if device == "cuda":
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def write_synthetic(directory, rows, seed):
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Data

    graph_dir = directory / "GRAPH_benchmark"
    fp_dir = directory / "FP_benchmark"
    graph_dir.mkdir()
    fp_dir.mkdir()
    rng = np.random.default_rng(seed)
    smiles = ["C" * (2 + i % 9) for i in range(rows)]
    pd.DataFrame({"SMILES": smiles, "target": np.arange(rows) % 2}).to_csv(
        directory / "dataset.csv", index=False)
    fingerprints = pd.DataFrame(rng.integers(0, 2, (rows, 2048), dtype=np.uint8),
                                columns=[f"Bit_{i + 1}" for i in range(2048)])
    fingerprints.insert(0, "SMILES", smiles)
    fingerprints.to_csv(fp_dir / "molecular_fingerprints.csv", index=False)
    (graph_dir / "smiles.txt").write_text("\n".join(smiles), encoding="utf-8")
    for i, molecule in enumerate(smiles):
        nodes = len(molecule)
        edges = torch.arange(nodes - 1)
        edge_index = torch.cat((torch.stack((edges, edges + 1)),
                                torch.stack((edges + 1, edges))), dim=1)
        torch.save(Data(x=torch.from_numpy(rng.random((nodes, 30), dtype=np.float32)),
                        edge_index=edge_index, edge_attr=torch.ones(2 * (nodes - 1), 11),
                        z=torch.full((nodes,), 6, dtype=torch.long),
                        pos=torch.from_numpy(rng.random((nodes, 3), dtype=np.float32)),
                        smiles=molecule), graph_dir / f"{i}.pt")


def build_model(dataset, mode, device):
    import torch
    from models import GCNModel, MLPModel

    if mode == "3d":
        from torch_geometric.nn import SchNet
        model = SchNet(hidden_channels=32, num_filters=32, num_interactions=2,
                       num_gaussians=25)
    else:
        # Inspect one prepared sample rather than depending on private cache fields.
        sample = dataset[0]
        fp = getattr(sample, "fp_prepared", None)
        fp = fp if fp is not None else getattr(sample, "fp", None)
        width = fp.shape[1] if fp is not None else 0
        if mode == "fingerprints":
            model = MLPModel(width, 32, 2, 16, 32, 1, dataset.num_outputs)
        else:
            model = GCNModel(dataset.num_node_features, 32, 2, 16, 0.0,
                             width, 32, 2, 16 if width else 0, 32, 1, dataset.num_outputs)
    return model.to(device=device, dtype=torch.float32)


def inputs_and_target(batch, mode, device):
    import torch

    target = getattr(batch, "y_prepared", None)
    target = target if target is not None else batch.y
    target = target.to(device=device, dtype=torch.float32, non_blocking=True)
    if mode == "3d":
        inputs = (batch.z.to(device=device, dtype=torch.long, non_blocking=True),
                  batch.pos.to(device=device, dtype=torch.float32, non_blocking=True),
                  batch.batch.to(device=device, non_blocking=True))
    else:
        fp = getattr(batch, "fp_prepared", None)
        fp = fp if fp is not None else getattr(batch, "fp", None)
        if fp is not None:
            fp = fp.to(device=device, dtype=torch.float32, non_blocking=True)
        inputs = ((fp,) if mode == "fingerprints" else
                  (batch.x.to(device=device, dtype=torch.float32, non_blocking=True),
                   batch.edge_index.to(device=device, dtype=torch.long, non_blocking=True),
                   batch.batch.to(device=device, non_blocking=True), fp))
    return inputs, target


def train_epoch(model, optimizer, loader, mode, device, task):
    import torch

    synchronize(device)
    started = time.perf_counter()
    batch_wait = 0.0
    transfer_host = compute_host = 0.0
    total = torch.zeros((), device=device)
    rows = 0
    iterator = iter(loader)
    while True:
        waiting = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_wait += time.perf_counter() - waiting
        transferring = time.perf_counter()
        inputs, target = inputs_and_target(batch, mode, device)
        transfer_host += time.perf_counter() - transferring
        computing = time.perf_counter()
        optimizer.zero_grad()
        predictions = model(*inputs).reshape_as(target)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(predictions, target)
                if task == "classification" else torch.nn.functional.mse_loss(predictions, target))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite diagnostic loss. Inspect targets and features.")
        loss.backward()
        optimizer.step()
        total += loss.detach() * batch.num_graphs
        rows += batch.num_graphs
        compute_host += time.perf_counter() - computing
    synchronize(device)
    elapsed = time.perf_counter() - started
    return dict(seconds=elapsed, loader_wait_seconds=batch_wait,
                transfer_host_seconds=transfer_host, compute_host_seconds=compute_host, rows=rows,
                rows_per_second=rows / elapsed, loss=float(total.cpu()) / rows)


def run_configuration(args, directory, workers, cache_mb):
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import Subset
    from datasets import MolecularDataset, MolecularDataset3D, prepare_dataset
    from gnn_runtime import make_loader

    torch.manual_seed(args.seed)
    graph_path = str(directory / args.graphs) if args.mode != "fingerprints" else None
    fp_path = str(directory / args.fingerprints) if args.mode in ("hybrid", "fingerprints") else None
    started = time.perf_counter()
    keywords = dict(target_datatype=torch.float64 if args.task == "regression" else torch.float32,
                    graph_cache_bytes=int(cache_mb * 1024**2))
    if args.mode == "3d":
        dataset = MolecularDataset3D(str(directory / args.dataset), graph_path, torch.float32, **keywords)
    else:
        inputs = {"hybrid": "Both", "graphs": "Graphs", "fingerprints": "Molecular fingerprints"}[args.mode]
        dataset = MolecularDataset(str(directory / args.dataset), inputs, graph_path, fp_path,
                                   torch.float32, **keywords)
    construction = time.perf_counter() - started
    if dataset.df.shape[1] < 2:
        raise ValueError("The training diagnostic requires at least one target column.")
    if args.task == "classification" and not np.isin(dataset.df.iloc[:, 1:].to_numpy(), [0, 1]).all():
        raise ValueError("Classification targets must be binary; use --task regression otherwise.")
    indices = np.random.default_rng(args.seed).choice(len(dataset), min(args.rows, len(dataset)),
                                                    replace=False).tolist()
    started = time.perf_counter()
    variance = pca = scaler = None
    if args.pca_components:
        if fp_path is None:
            raise ValueError("PCA requires fingerprints.")
        variance = VarianceThreshold()
        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        pca.fit(variance.fit_transform(dataset.fingerprint_values[indices]))
    if args.task == "regression":
        scaler = MinMaxScaler().fit(dataset.df.iloc[indices, 1:].to_numpy())
    dataset.clear_cache()
    prepare_dataset(dataset, variance, pca, scaler)
    preparation = time.perf_counter() - started
    loader = make_loader(Subset(dataset, indices), args.batch_size, seed=args.seed,
                         loader_workers=workers, pin_memory=args.pin_memory)
    passes = []
    for _ in range(args.passes):
        started = time.perf_counter()
        rows = sum(batch.num_graphs for batch in loader)
        elapsed = time.perf_counter() - started
        passes.append(dict(seconds=elapsed, rows=rows, rows_per_second=rows / elapsed))
    after_iteration = snapshot(args.device)
    cache_info = dataset.graph_cache_info
    model = build_model(dataset, args.mode, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    epochs = []
    for _ in range(args.epochs):
        epochs.append(train_epoch(model, optimizer, loader, args.mode, args.device, args.task))
        scheduler.step()
    references = dict(dataset=weakref.ref(dataset), loader=weakref.ref(loader),
                      model=weakref.ref(model), optimizer=weakref.ref(optimizer),
                      scheduler=weakref.ref(scheduler), parameter=weakref.ref(next(model.parameters())))
    parameter = next(model.parameters())
    if parameter.grad is not None:
        references["gradient"] = weakref.ref(parameter.grad)
    if "exp_avg" in optimizer.state.get(parameter, {}):
        references["adam_moment"] = weakref.ref(optimizer.state[parameter]["exp_avg"])
    return dict(workers=workers, graph_cache_mib=cache_mb, pin_memory=args.pin_memory,
                dataset_rows=len(dataset), sampled_rows=len(indices), construction_seconds=construction,
                preprocessing_seconds=preparation, iteration_passes=passes, graph_cache=cache_info,
                training_epochs=epochs, after_iteration=after_iteration,
                after_training=snapshot(args.device)), references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, help="Existing working directory; omitted uses disposable synthetic graphs.")
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--graphs", default="GRAPH_benchmark")
    parser.add_argument("--fingerprints", default="FP_benchmark")
    parser.add_argument("--mode", choices=("hybrid", "graphs", "fingerprints", "3d"), default="hybrid")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=1, help="Repeat complete dataset/model creation, training, and release.")
    parser.add_argument("--workers", type=int, choices=(0, 2, 4), nargs="+", default=[0])
    parser.add_argument("--cache-mb", type=float, nargs="+", default=[128], help="Graph cache budgets in MiB; 0 disables caching.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; native pool environment variables are respected.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Also save JSON to a new file; existing files are never overwritten.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Run this diagnostic with the project's Python 3.11 environment.")
    if min(args.rows, args.batch_size, args.passes, args.epochs, args.cycles, args.threads) < 1:
        parser.error("Rows, batch size, passes, epochs, cycles, and threads must be positive.")
    if any(not math.isfinite(value) or value < 0 for value in args.cache_mb) or args.pca_components < 0:
        parser.error("Cache budgets and PCA components cannot be negative.")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report filename.")
    import torch
    import torch_geometric
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this process; use --device cpu or check the runtime.")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="gnn-benchmark-") as temporary:
        directory = args.workdir.resolve() if args.workdir else Path(temporary)
        if args.workdir is None:
            write_synthetic(directory, args.rows, args.seed)
        report = dict(python=platform.python_version(), platform=platform.platform(),
                      torch=torch.__version__, pyg=torch_geometric.__version__,
                      cuda_available=torch.cuda.is_available(), device=args.device,
                      cuda_device=torch.cuda.get_device_name() if args.device == "cuda" else None,
                      cuda_runtime=torch.version.cuda,
                      source=str(directory) if args.workdir else "temporary synthetic features",
                      mode=args.mode, task=args.task, seed=args.seed, batch_size=args.batch_size,
                      torch_threads=torch.get_num_threads(), pca_components=args.pca_components,
                      native_thread_limits={name: os.environ.get(name) for name in
                                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                      before=snapshot(args.device), runs=[])
        for workers in args.workers:
            for cache_mb in args.cache_mb:
                for cycle in range(args.cycles):
                    result, references = run_configuration(args, directory, workers, cache_mb)
                    gc.collect()  # Diagnostic boundary only; never inside production batches.
                    synchronize(args.device)
                    result.update(cycle=cycle + 1, released={name: ref() is None for name, ref in references.items()},
                                  after_release=snapshot(args.device))
                    report["runs"].append(result)
        report["all_objects_released"] = all(all(run["released"].values()) for run in report["runs"])
        text = json.dumps(report, indent=2, allow_nan=False)
        print(text)
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text + "\n")
        return 0 if report["all_objects_released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
