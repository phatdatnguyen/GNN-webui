import os
import re
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, SaltRemover, rdFingerprintGenerator, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
import deepchem as dc
import torch
from torch_geometric.data import Data

# Anchor all user data to the repo directory so the app behaves the same regardless of the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

def sanitize_path_component(name, label="name"):
    """Validate that a user-supplied string is a single safe path component.

    The working directory, dataset, checkpoint and export names all come straight from
    free-text boxes, so without this a value like '../..' or '/etc' would escape Data/.
    This rejects rather than rewrites: silently turning '/etc' into 'etc' would put the
    user's data somewhere they did not ask for.
    """
    if name is None:
        raise ValueError(f"Please specify a {label}.")
    component = str(name).strip()
    if component == "":
        raise ValueError(f"Please specify a {label}.")
    if component in (".", ".."):
        raise ValueError(f"Invalid {label}: {name!r}.")
    if not re.fullmatch(r"[A-Za-z0-9._ -]+", component):
        raise ValueError(f"Invalid {label}: {name!r}. Use only letters, digits, spaces, "
                         "'.', '_' or '-' - path separators are not allowed.")
    return component

def export_result_csv(working_directory_path, file_name, frame):
    """Write a nonempty result table inside the opened working directory."""
    if not working_directory_path or not os.path.isdir(working_directory_path):
        raise ValueError("Please open an existing working directory first.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Please produce results before exporting.")
    file_name = sanitize_path_component(file_name, "export file name")
    filename = file_name if file_name.lower().endswith(".csv") else f"{file_name}.csv"
    frame.to_csv(os.path.join(working_directory_path, filename), index=False)
    return filename

def list_graph_files(graph_directory_path):
    """Return the graph shards written by extract_graphs, ordered by their numeric index.

    Only files named '<int>.pt' count - anything else a user drops in the directory must not
    change the dataset length.
    """
    if not os.path.isdir(graph_directory_path):
        return []
    indices = sorted(
        int(f[:-3]) for f in os.listdir(graph_directory_path)
        if re.fullmatch(r"(0|[1-9][0-9]*)\.pt", f)
        and os.path.isfile(os.path.join(graph_directory_path, f))
    )
    return indices

# Written alongside the graph shards so the dataset can verify alignment against the
# dataset CSV completely, instead of spot-checking a few graphs.
GRAPH_MANIFEST = "smiles.txt"
FINGERPRINT_CONFIG = "fingerprint_config.json"
FINGERPRINT_NAMES = [
    "Morgan fingerprint", "Atom-pair fingerprint", "Topological-torsion fingerprint",
    "Avalon fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint",
]

def write_graph_manifest(graph_directory_path, smiles_list):
    with open(os.path.join(graph_directory_path, GRAPH_MANIFEST), "w") as handle:
        handle.write("\n".join(smiles_list))

def read_graph_manifest(graph_directory_path):
    """Return the SMILES the graphs were built from, or None for pre-manifest extractions."""
    path = os.path.join(graph_directory_path, GRAPH_MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        content = handle.read()
    return content.split("\n") if content else []

def split_molecular_dataset(dataset, test_ratio, val_ratio, random_seed, split_method="Random"):
    """Return train/validation/test subsets using random rows or whole Murcko scaffolds.

    Holdout sizes use the same flooring as the original random split. Scaffold sizes
    approximate those targets: largest groups go to the greatest remaining deficit,
    with seeded ordering of equal-size groups. Reserve groups for any empty requested
    splits. Salt stripping matches extraction; chirality is ignored, and all acyclic
    molecules share the empty scaffold. No rows are dropped or reordered on disk.
    """
    total_len = len(dataset)
    if (not np.isfinite(test_ratio) or not np.isfinite(val_ratio)
            or not 0 <= test_ratio <= 1 or not 0 <= val_ratio <= 1
            or test_ratio + val_ratio > 1):
        raise ValueError("Invalid split sizes. Please adjust test/val sizes.")
    test_size = int(total_len * test_ratio)
    val_size = int(total_len * val_ratio)
    train_size = total_len - test_size - val_size
    if train_size <= 0:
        raise ValueError("Invalid split sizes: the training split must contain at least one molecule.")
    split_sizes = [train_size, val_size, test_size]
    generator = torch.Generator().manual_seed(random_seed)
    if split_method == "Random":
        return tuple(torch.utils.data.random_split(dataset, split_sizes, generator=generator))
    if split_method != "Scaffold":
        raise ValueError(f"Unknown split method '{split_method}'. Please select Random or Scaffold.")

    salt_remover = SaltRemover.SaltRemover()
    scaffolds = {}
    smiles_scaffolds = {}
    for index, smiles in enumerate(dataset.df['SMILES']):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Invalid SMILES at dataset row {index}: {smiles!r}. "
                             "Please correct the dataset before scaffold splitting.")
        if smiles not in smiles_scaffolds:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or mol.GetNumAtoms() == 0:
                    raise ValueError("SMILES could not be parsed.")
                mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
                smiles_scaffolds[smiles] = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            except Exception as exc:
                raise ValueError(f"Invalid SMILES at dataset row {index}: {smiles!r}. "
                                 "Please correct the dataset before scaffold splitting.") from exc
        scaffolds.setdefault(smiles_scaffolds[smiles], []).append(index)

    active_splits = [index for index, size in enumerate(split_sizes) if size > 0]
    groups = list(scaffolds.values())
    if len(groups) < len(active_splits):
        raise ValueError(
            f"Scaffold splitting found {len(groups)} distinct scaffold group(s), but "
            f"{len(active_splits)} nonempty splits were requested. Reduce the holdout sizes "
            "or choose Random. All acyclic molecules share one scaffold.")
    # Use a local generator so choosing a split does not change model/training RNG state.
    groups = [groups[index] for index in torch.randperm(len(groups), generator=generator).tolist()]
    groups.sort(key=len, reverse=True)
    split_indices = [[], [], []]
    for position, group in enumerate(groups):
        empty_splits = [index for index in active_splits if not split_indices[index]]
        candidates = empty_splits if len(groups) - position == len(empty_splits) else active_splits
        destination = max(candidates, key=lambda index: split_sizes[index] - len(split_indices[index]))
        split_indices[destination].extend(group)
    return tuple(torch.utils.data.Subset(dataset, sorted(indices)) for indices in split_indices)


def ensure_optimizer_matches_model(model, optimizer):
    """Guard against training with an optimizer built for a previous model.

    The optimizer captures model.parameters() when it is created. If the user creates a new
    model afterwards, backward() fills gradients on the new model while step() walks the old
    parameter list - so nothing is updated and the loss sits perfectly flat.
    """
    if model is None:
        raise ValueError("Please create a model first.")
    if optimizer is None:
        raise ValueError("Please create an optimizer first.")
    model_parameters = {id(p) for p in model.parameters()}
    optimizer_parameters = {id(p) for group in optimizer.param_groups for p in group['params']}
    if model_parameters != optimizer_parameters:
        raise ValueError("The optimizer belongs to a previously created model. "
                         "Please click 'Create optimizer' again before training.")

def ensure_model_on_device(model, device):
    """The model is placed on a device once, at creation; the dropdown is read every run."""
    if model is None:
        raise ValueError("Please create a model first.")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return
    if model_device.type != torch.device(device).type:
        raise ValueError(f"The model is on '{model_device.type}' but the Device setting is "
                         f"'{device}'. Re-create the model to move it to '{device}'.")

def load_model_checkpoint(model, checkpoint_file_path):
    """Validate checkpoint compatibility before changing any current model weights."""
    # Loading onto CPU also permits opening CUDA checkpoints on machines without CUDA.
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu', weights_only=False)
    saved_class = checkpoint.get('model_class')
    if saved_class is not None and saved_class != type(model).__name__:
        raise ValueError(f"This checkpoint was saved from a {saved_class} but the current model is a "
                         f"{type(model).__name__}. Re-create the model with the original architecture first.")

    saved_state = checkpoint['state_dict']
    current_state = model.state_dict()
    # load_state_dict copies compatible tensors before reporting missing keys or shape
    # mismatches. Preflight the entire state so a rejected load cannot mix two models.
    if saved_state.keys() != current_state.keys():
        raise ValueError("Checkpoint parameters do not match the current model. "
                         "Re-create the model with the original architecture first.")
    for name, value in current_state.items():
        saved_value = saved_state[name]
        if (not isinstance(saved_value, torch.Tensor)
                or saved_value.shape != value.shape
                or saved_value.layout != value.layout):
            raise ValueError(f"Checkpoint parameter '{name}' does not match the current model. "
                             "Re-create the model with the original architecture first.")
    model.load_state_dict(saved_state)
    return checkpoint

def process_dataset_file(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_column_names, is_3d_dataset=False):
    try:
        dataset_name = sanitize_path_component(dataset_name, "dataset name")

        # Load the dataset and validate columns
        df = pd.read_csv(dataset_file_path)
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in the dataset.")
        
        # Validate target columns
        if target_column_names == "":
            target_column_name_list = []
        else:
            target_column_name_list = [col.strip() for col in target_column_names.split(",")]
            if is_3d_dataset and len(target_column_name_list) > 1:
                raise ValueError(f"3D model only accept 1 target column")
            missing_target_columns = [col for col in target_column_name_list if col not in df.columns]
            if missing_target_columns:
                raise ValueError(f"Target columns not found in the dataset: {', '.join(missing_target_columns)}")
        
        df = df[[smiles_column] + target_column_name_list] # Keep only the SMILES and target columns (if there are any target columns)
        df = df.rename(columns={smiles_column: 'SMILES'}) # Rename the SMILES column to a standard name expected by the rest of the code
        df = df.dropna() # Remove the rows with no data

        # Filter out invalid SMILES and those with no carbon.
        # Build a mask in one pass: filtering the frame per rejected row is quadratic, and
        # dropping by value would also take out unrelated duplicates of the same SMILES.
        salt_remover = SaltRemover.SaltRemover()
        verdicts = {}
        keep = []
        for smiles in df['SMILES']:
            if smiles not in verdicts:
                mol_obj = Chem.MolFromSmiles(smiles)
                if mol_obj is None:
                    verdicts[smiles] = False
                    print(f"Invalid SMILES '{smiles}' removed from the dataset.")
                else:
                    mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                    carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
                    verdicts[smiles] = carbon_count > 0 and smiles != '[C]'
                    if not verdicts[smiles]:
                        print(f"SMILES '{smiles}' with no carbon atom removed from the dataset.")
            keep.append(verdicts[smiles])
        df = df.loc[keep].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid molecules remain after filtering the dataset.")

        # Save the uploaded dataset to the working directory
        save_path = os.path.join(working_directory_path, f"{dataset_name}.csv")
        df.to_csv(save_path, index=False)

        status = f"Dataset '{dataset_name}' uploaded successfully to working directory."
        return f"<span style='color: green;'>{status}</span>", df
    except Exception as exc:
        status = f"Error uploading dataset: {exc}"
        return f"<span style='color: red;'>{status}</span>", None

def _publish_extraction(staged_directory, output_directory, dataset_file_path, df, successful_indices):
    """Publish completed features, retaining the previous data if extraction or saving fails."""
    if not successful_indices:
        raise ValueError("No molecules were extracted successfully. The dataset and previous "
                         "extracted features have been preserved.")
    if os.path.lexists(output_directory) and (
            os.path.islink(output_directory) or not os.path.isdir(output_directory)):
        raise ValueError("The extraction destination must be a directory, not a file or symbolic link.")

    staging_root = os.path.dirname(staged_directory)
    filtered_dataset = None
    if len(successful_indices) != len(df):
        filtered_dataset = os.path.join(staging_root, "dataset.csv")
        df.iloc[successful_indices].reset_index(drop=True).to_csv(filtered_dataset, index=False)

    had_previous_output = os.path.exists(output_directory)
    backup_root = None
    backup_directory = None
    if had_previous_output:
        # Keep the only copy of previous features outside TemporaryDirectory: a failed
        # restoration must survive the caller cleaning up the new extraction.
        backup_root = tempfile.mkdtemp(
            prefix=f".{os.path.basename(output_directory)}-backup-",
            dir=os.path.dirname(output_directory))
        backup_directory = os.path.join(backup_root, "previous")
    try:
        if had_previous_output:
            os.replace(output_directory, backup_directory)
        os.replace(staged_directory, output_directory)
        if filtered_dataset is not None:
            os.replace(filtered_dataset, dataset_file_path)
    except BaseException as publish_error:
        # An interrupt can arrive after a rename completed but before Python regains
        # control. Inspect the paths instead of relying on a flag set after os.replace.
        # Once the filtered CSV has moved, keep its matching new features in place.
        if filtered_dataset is not None and not os.path.exists(filtered_dataset):
            raise
        try:
            if backup_directory is not None and os.path.exists(backup_directory):
                if os.path.exists(output_directory):
                    shutil.rmtree(output_directory)
                os.replace(backup_directory, output_directory)
            elif not had_previous_output and not os.path.exists(staged_directory):
                shutil.rmtree(output_directory)
        except BaseException as restore_error:
            if backup_directory is None:
                raise RuntimeError(
                    f"Extraction publication failed ({publish_error}) and new features "
                    f"could not be removed from '{output_directory}' ({restore_error}).") from restore_error
            preserved_directory = (backup_directory if os.path.exists(backup_directory)
                                   else output_directory)
            raise RuntimeError(
                f"Extraction publication failed ({publish_error}) and previous features "
                f"could not be restored ({restore_error}). The backup is preserved at "
                f"'{preserved_directory}'.") from restore_error
        if backup_root is not None:
            os.rmdir(backup_root)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)


def extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress):
    try:
        # This path is passed to shutil.rmtree below, so it must be a single safe component.
        graph_directory = sanitize_path_component(graph_directory, "graph directory")
        dataset_file_name = sanitize_path_component(dataset_file_name, "dataset file name")
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        output_directory = os.path.join(working_directory_path, graph_directory)
        with tempfile.TemporaryDirectory(prefix=".extraction-", dir=working_directory_path) as staging_root:
            graph_dir_path = os.path.join(staging_root, "features")
            os.makedirs(graph_dir_path)

            # Graph extraction and featurization
            if graph_featurizer_name == 'MolGraphConvFeaturizer':
                graph_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
            elif graph_featurizer_name == 'PagtnMolGraphFeaturizer':
                graph_featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
            else:
                graph_featurizer = dc.feat.DMPNNFeaturizer(is_adding_hs=False)
            salt_remover = SaltRemover.SaltRemover()

            successful_indices = []
            graph_index = 0
            for i in progress.tqdm(range(len(smiles_list)), total=len(smiles_list), desc="Extracting graphs"):
                smiles = smiles_list[i]
                try:
                    mol_obj = Chem.MolFromSmiles(smiles)
                    mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                    mol_obj = Chem.AddHs(mol_obj)
                    graph = graph_featurizer.featurize(mol_obj)
                    node_feats = torch.tensor(np.array(graph[0].node_features), dtype=datatype)
                    # edge_index holds node indices, not features: it must stay integral. Storing it as
                    # a low-precision float silently corrupts indices above 2048 once batches are collated.
                    edge_index = torch.tensor(np.array(graph[0].edge_index), dtype=torch.long)
                    edge_attr = torch.tensor(np.array(graph[0].edge_features), dtype=datatype)

                    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, smiles=df['SMILES'].iloc[i])

                except Exception as exc:
                    print(f"Error processing SMILES '{smiles}': {exc}")
                    continue

                # A storage failure must abort extraction, not remove a valid molecule.
                torch.save(data, os.path.join(graph_dir_path, f'{graph_index}.pt'))
                successful_indices.append(i)
                graph_index += 1

            graph_count = graph_index
            n_failed = len(smiles_list) - graph_count
            write_graph_manifest(graph_dir_path, [smiles_list[i] for i in successful_indices])
            _publish_extraction(graph_dir_path, output_directory, dataset_file_path, df, successful_indices)
            if n_failed > 0:
                status = f"{graph_count} graphs extracted to '{graph_directory}'; {n_failed} molecule(s) failed and were removed from {dataset_file_name}. Re-run molecular fingerprint extraction if previously done."
            else:
                status = f"{graph_count} graphs extracted and saved to '{graph_directory}' successfully."
            return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        return f"<span style='color: red;'>Error extracting graphs: {exc}</span>"

def extract_3d_graphs(working_directory_path, dataset_file_name, graph_directory, num_conformers, force_field, datatype, progress):
    try:
        # This path is passed to shutil.rmtree below, so it must be a single safe component.
        graph_directory = sanitize_path_component(graph_directory, "graph directory")
        dataset_file_name = sanitize_path_component(dataset_file_name, "dataset file name")
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        output_directory = os.path.join(working_directory_path, graph_directory)
        with tempfile.TemporaryDirectory(prefix=".extraction-", dir=working_directory_path) as staging_root:
            graph_dir_path = os.path.join(staging_root, "features")
            os.makedirs(graph_dir_path)

            # Graph extraction and featurization
            salt_remover = SaltRemover.SaltRemover()

            successful_indices = []
            graph_index = 0
            for i in progress.tqdm(range(len(smiles_list)), total=len(smiles_list), desc="Extracting graphs"):
                smiles = smiles_list[i]
                try:
                    mol_obj = Chem.MolFromSmiles(smiles)
                    mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                    mol_obj = Chem.AddHs(mol_obj)
                    params = AllChem.ETKDGv3()
                    ids = AllChem.EmbedMultipleConfs(mol_obj, numConfs=num_conformers, params=params)
                    if force_field == "MMFF":
                        if not AllChem.MMFFHasAllMoleculeParams(mol_obj):
                            raise ValueError("MMFF parameters are not available for this molecule.")
                        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_obj)
                        if mmff_props is None:
                            raise ValueError("MMFF parameters are not available for this molecule.")
                    energies = []
                    for conf_id in ids:
                        if force_field == "MMFF":
                            AllChem.MMFFOptimizeMolecule(mol_obj, confId=conf_id, maxIters=200)
                            ff = AllChem.MMFFGetMoleculeForceField(mol_obj, mmff_props, confId=conf_id)
                        else: #force_field=="UFF"
                            AllChem.UFFOptimizeMolecule(mol_obj, confId=conf_id, maxIters=200)
                            ff = AllChem.UFFGetMoleculeForceField(mol_obj, confId=conf_id)
                        energy = ff.CalcEnergy()
                        energies.append((conf_id, energy))
                    if not energies:
                        raise ValueError("No conformers could be embedded for this molecule.")
                    min_conf_id = min(energies, key=lambda x: x[1])[0]
                    min_conf = mol_obj.GetConformer(id=min_conf_id)
                    z = [atom.GetAtomicNum() for atom in mol_obj.GetAtoms()]
                    pos = np.array([
                        [
                            min_conf.GetAtomPosition(i).x,
                            min_conf.GetAtomPosition(i).y,
                            min_conf.GetAtomPosition(i).z,
                        ]
                        for i in range(mol_obj.GetNumAtoms())
                    ])
                    # Atomic numbers index an embedding table, so they must stay integral.
                    z = torch.tensor(z, dtype=torch.long)
                    pos = torch.tensor(pos, dtype=datatype)

                    data = Data(z=z, pos=pos, smiles=df['SMILES'].iloc[i])

                except Exception as exc:
                    print(f"Error processing SMILES '{smiles}': {exc}")
                    continue

                # A storage failure must abort extraction, not remove a valid molecule.
                torch.save(data, os.path.join(graph_dir_path, f'{graph_index}.pt'))
                successful_indices.append(i)
                graph_index += 1

            graph_count = graph_index
            n_failed = len(smiles_list) - graph_count
            write_graph_manifest(graph_dir_path, [smiles_list[i] for i in successful_indices])
            _publish_extraction(graph_dir_path, output_directory, dataset_file_path, df, successful_indices)
            if n_failed > 0:
                status = f"{graph_count} graphs extracted to '{graph_directory}'; {n_failed} molecule(s) failed conformer generation and were removed from {dataset_file_name}."
            else:
                status = f"{graph_count} graphs extracted and saved to '{graph_directory}' successfully."
            return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        return f"<span style='color: red;'>Error extracting graphs: {exc}</span>"
    
def extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress):
    try:
        if molecular_fingerprint_name not in FINGERPRINT_NAMES:
            raise ValueError("Please select a supported molecular fingerprint.")
        if int(number_of_bits) != number_of_bits or number_of_bits <= 0:
            raise ValueError("The number of fingerprint bits must be a positive integer.")
        number_of_bits = int(number_of_bits)
        if molecular_fingerprint_name == "Morgan fingerprint":
            if int(radius) != radius or radius < 0:
                raise ValueError("The Morgan radius must be a nonnegative integer.")
            radius = int(radius)
        # This path is passed to shutil.rmtree below, so it must be a single safe component.
        molecular_fingerprint_directory = sanitize_path_component(molecular_fingerprint_directory, "molecular fingerprint directory")
        dataset_file_name = sanitize_path_component(dataset_file_name, "dataset file name")
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        output_directory = os.path.join(working_directory_path, molecular_fingerprint_directory)
        with tempfile.TemporaryDirectory(prefix=".extraction-", dir=working_directory_path) as staging_root:
            fingerprint_dir_path = os.path.join(staging_root, "features")
            os.makedirs(fingerprint_dir_path)

            # Molecular fingerprint extraction
            fp_list = []
            successful_indices = []
            salt_remover = SaltRemover.SaltRemover()
            for i in progress.tqdm(range(len(smiles_list)), total=len(smiles_list), desc="Extracting molecular fingerprints"):
                smiles = smiles_list[i]
                try:
                    mol_obj = Chem.MolFromSmiles(smiles)
                    mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                    mol_obj = Chem.AddHs(mol_obj)

                    if molecular_fingerprint_name == 'Morgan fingerprint':
                        fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=number_of_bits).GetFingerprint(mol_obj)
                    elif molecular_fingerprint_name == 'Atom-pair fingerprint':
                        fp = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=number_of_bits).GetFingerprintAsNumPy(mol_obj)
                    elif molecular_fingerprint_name == 'Topological-torsion fingerprint':
                        fp = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=number_of_bits).GetFingerprintAsNumPy(mol_obj)
                    elif molecular_fingerprint_name == 'Avalon fingerprint':
                        fp = np.array(pyAvalonTools.GetAvalonFP(mol_obj, nBits=number_of_bits))
                    elif molecular_fingerprint_name == 'Layered fingerprint':
                        fp = np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=number_of_bits))
                    elif molecular_fingerprint_name == 'Pattern fingerprint':
                        fp = np.array(rdmolops.PatternFingerprint(mol_obj, fpSize=number_of_bits))
                    else: # if self.molecular_fingerprint_name == 'RDKit fingerprint':
                        fp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=number_of_bits).GetFingerprintAsNumPy(mol_obj)

                    row = [smiles]
                    row.extend(fp)
                    fp_list.append(row)
                    successful_indices.append(i)
                except Exception as exc:
                    print(f"Error processing SMILES '{smiles}': {exc}")
                    continue

            # Save the molecular fingerprints to a CSV file
            column_names = ["SMILES"] + [f'Bit_{i}' for i in range(1, number_of_bits + 1)]
            pd.DataFrame(fp_list, columns=column_names).to_csv(
                os.path.join(fingerprint_dir_path, 'molecular_fingerprints.csv'), index=False)
            # Publish the featurizer settings with the bits, so saved classical models
            # can reject different fingerprint algorithms even when their widths match.
            with open(os.path.join(fingerprint_dir_path, FINGERPRINT_CONFIG), "w") as handle:
                json.dump({"name": molecular_fingerprint_name,
                           "radius": radius if molecular_fingerprint_name == "Morgan fingerprint" else None,
                           "number_of_bits": number_of_bits}, handle)

            n_extracted = len(fp_list)
            _publish_extraction(fingerprint_dir_path, output_directory, dataset_file_path, df, successful_indices)
            n_failed = len(smiles_list) - n_extracted
            if n_failed > 0:
                status = f"{n_extracted} molecular fingerprints extracted to '{molecular_fingerprint_directory}'; {n_failed} molecule(s) failed and were removed from {dataset_file_name}. Re-run graph extraction if previously done."
            else:
                status = f"{n_extracted} molecular fingerprints extracted and saved to '{molecular_fingerprint_directory}' successfully."
            return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        return f"<span style='color: red;'>Error extracting molecular fingerprints: {exc}</span>"
