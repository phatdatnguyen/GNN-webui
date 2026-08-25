import os
import re
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, SaltRemover, rdFingerprintGenerator, rdmolops
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

def list_graph_files(graph_directory_path):
    """Return the graph shards written by extract_graphs, ordered by their numeric index.

    Only files named '<int>.pt' count - anything else a user drops in the directory must not
    change the dataset length.
    """
    if not os.path.isdir(graph_directory_path):
        return []
    indices = sorted(
        int(f[:-3]) for f in os.listdir(graph_directory_path)
        if f.endswith(".pt") and f[:-3].isdigit()
    )
    return indices

# Written alongside the graph shards so the dataset can verify alignment against the
# dataset CSV completely, instead of spot-checking a few graphs.
GRAPH_MANIFEST = "smiles.txt"

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
        df = df[keep].reset_index(drop=True)

        # Save the uploaded dataset to the working directory
        save_path = os.path.join(working_directory_path, f"{dataset_name}.csv")
        df.to_csv(save_path, index=False)

        status = f"Dataset '{dataset_name}' uploaded successfully to working directory."
        return f"<span style='color: green;'>{status}</span>", df
    except Exception as exc:
        status = f"Error uploading dataset: {exc}"
        return f"<span style='color: red;'>{status}</span>", None

def extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name, datatype, progress):
    try:
        # This path is passed to shutil.rmtree below, so it must be a single safe component.
        graph_directory = sanitize_path_component(graph_directory, "graph directory")
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        graph_dir_path = os.path.join(working_directory_path, graph_directory)
        if os.path.exists(graph_dir_path):
            shutil.rmtree(graph_dir_path)
        os.makedirs(graph_dir_path, exist_ok=True)

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

                # Save the graph data
                torch.save(data, os.path.join(graph_dir_path, f'{graph_index}.pt'))
                successful_indices.append(i)
                graph_index += 1
            except Exception as exc:
                print(f"Error processing SMILES '{smiles}': {exc}")
                continue

        graph_count = graph_index
        n_failed = len(smiles_list) - graph_count
        write_graph_manifest(graph_dir_path, [smiles_list[i] for i in successful_indices])
        if n_failed > 0:
            df.iloc[successful_indices].reset_index(drop=True).to_csv(dataset_file_path, index=False)
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
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        graph_dir_path = os.path.join(working_directory_path, graph_directory)
        if os.path.exists(graph_dir_path):
            shutil.rmtree(graph_dir_path)
        os.makedirs(graph_dir_path, exist_ok=True)

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

                # Save the graph data
                torch.save(data, os.path.join(graph_dir_path, f'{graph_index}.pt'))
                successful_indices.append(i)
                graph_index += 1
            except Exception as exc:
                print(f"Error processing SMILES '{smiles}': {exc}")
                continue

        graph_count = graph_index
        n_failed = len(smiles_list) - graph_count
        write_graph_manifest(graph_dir_path, [smiles_list[i] for i in successful_indices])
        if n_failed > 0:
            df.iloc[successful_indices].reset_index(drop=True).to_csv(dataset_file_path, index=False)
            status = f"{graph_count} graphs extracted to '{graph_directory}'; {n_failed} molecule(s) failed conformer generation and were removed from {dataset_file_name}."
        else:
            status = f"{graph_count} graphs extracted and saved to '{graph_directory}' successfully."
        return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        return f"<span style='color: red;'>Error extracting graphs: {exc}</span>"
    
def extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress):
    try:
        # This path is passed to shutil.rmtree below, so it must be a single safe component.
        molecular_fingerprint_directory = sanitize_path_component(molecular_fingerprint_directory, "molecular fingerprint directory")
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        fingerprint_dir_path = os.path.join(working_directory_path, molecular_fingerprint_directory)
        if os.path.exists(fingerprint_dir_path):
            shutil.rmtree(fingerprint_dir_path)
        os.makedirs(fingerprint_dir_path, exist_ok=True)

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
        with open(os.path.join(fingerprint_dir_path, 'molecular_fingerprints.csv'), 'w') as f:
            f.write(','.join(column_names) + '\n')
            for row in fp_list:
                f.write(','.join(map(str, row)) + '\n')

        n_extracted = len(fp_list)
        n_failed = len(smiles_list) - n_extracted
        if n_failed > 0:
            df.iloc[successful_indices].reset_index(drop=True).to_csv(dataset_file_path, index=False)
            status = f"{n_extracted} molecular fingerprints extracted to '{molecular_fingerprint_directory}'; {n_failed} molecule(s) failed and were removed from {dataset_file_name}. Re-run graph extraction if previously done."
        else:
            status = f"{n_extracted} molecular fingerprints extracted and saved to '{molecular_fingerprint_directory}' successfully."
        return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        return f"<span style='color: red;'>Error extracting molecular fingerprints: {exc}</span>"
