import os
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import SaltRemover, rdFingerprintGenerator, rdmolops
import deepchem as dc
import torch
from torch_geometric.data import Data

def process_dataset_file(dataset_file_path, working_directory_path, dataset_name, smiles_column, target_column_names):
    try:
        # Load the dataset and validate columns
        df = pd.read_csv(dataset_file_path)
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in the dataset.")
        
        # Validate target columns
        if target_column_names == "":
            target_column_name_list = []
        else:
            target_column_name_list = [col.strip() for col in target_column_names.split(",")]
            missing_target_columns = [col for col in target_column_name_list if col not in df.columns]
            if missing_target_columns:
                raise ValueError(f"Target columns not found in the dataset: {', '.join(missing_target_columns)}")
        
        df = df[[smiles_column] + target_column_name_list] # Keep only the SMILES and target columns (if there are any target columns)
        df = df.rename(columns={smiles_column: 'SMILES'}) # Rename the SMILES column to a standard name expected by the rest of the code
        df = df.dropna() # Remove the rows with no data

        # Filter out invalid SMILES and those with no carbon or only 1 carbon
        for smiles in df['SMILES']:
            mol_obj = Chem.MolFromSmiles(smiles)
            if mol_obj is None:
                df = df[df['SMILES'] != smiles]
                print(f"Invalid SMILES '{smiles}' removed from the dataset.")
                continue
            salt_remover = SaltRemover.SaltRemover()
            mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
            carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
            if carbon_count == 0 or smiles == '[C]':
                df = df[df['SMILES'] != smiles]
                print(f"SMILES '{smiles}' with no carbon atom removed from the dataset.")

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

        graph_index = 0
        for i in progress.tqdm(range(len(smiles_list)), total=len(smiles_list), desc="Extracting graphs"):
            smiles = smiles_list[i]
            try:
                mol_obj = Chem.MolFromSmiles(smiles)
                mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                mol_obj = Chem.AddHs(mol_obj)
                graph = graph_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(graph[0].node_features), dtype=datatype)
                edge_index = torch.tensor(np.array(graph[0].edge_index), dtype=datatype)
                edge_attr = torch.tensor(np.array(graph[0].edge_features), dtype=datatype)

                data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, smiles=df['SMILES'].iloc[i])

                # Save the graph data
                torch.save(data, os.path.join(graph_dir_path, f'{graph_index}.pt'))
                graph_index += 1
            except Exception as exc:
                print(f"Error processing SMILES '{smiles}': {exc}")
                continue

        graph_count = graph_index
        status = f"{graph_count} graphs extracted and saved to '{graph_directory}' successfully."
        return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        status = f"Error extracting graphs: {exc}"
        return f"<span style='color: red;'>Error extracting graphs: {exc}</span>"
    
def extract_molecule_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory, molecular_fingerprint_name, radius, number_of_bits, progress):
    try:
        dataset_file_path = os.path.join(working_directory_path, dataset_file_name)
        df = pd.read_csv(dataset_file_path)
        smiles_list = df['SMILES'].tolist()
        fingerprint_dir_path = os.path.join(working_directory_path, molecular_fingerprint_directory)
        if os.path.exists(fingerprint_dir_path):
            shutil.rmtree(fingerprint_dir_path)
        os.makedirs(fingerprint_dir_path, exist_ok=True)

        # Molecular fingerprint extraction
        fp_list = []
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
            except:
                continue

        # Save the molecular fingerprints to a CSV file
        column_names = ["SMILES"] + [f'Bit_{i}' for i in range(1, number_of_bits + 1)]
        with open(os.path.join(fingerprint_dir_path, 'molecular_fingerprints.csv'), 'w') as f:
            f.write(','.join(column_names) + '\n')
            for row in fp_list:
                f.write(','.join(map(str, row)) + '\n')

        status = f"{len(smiles_list)} molecular fingerprints extracted and saved to '{molecular_fingerprint_directory}' successfully."
        return f"<span style='color: green;'>{status}</span>"
    except Exception as exc:
        status = f"Error extracting molecular fingerprints: {exc}"
        return f"<span style='color: red;'>{status}</span>"