# Import necessary libraries
import pandas as pd
from rdkit import Chem

# Read the CSV file into a pandas DataFrame
# Replace 'fluorophores.csv' with the path to your CSV file
# Ensure the CSV has a column named 'SMILES' containing the SMILES strings
df = pd.read_csv('nchromo_ems.csv')

# Define patterns for each fluorophore class
fluorophore_classes = {
    'Coumarin': ['O=C1C=CC2=C(O1)C=CC=C2', 'O=c1ccc(cccc2)c2o1'],
    'Fluorescein': ['Oc1cc2oc3cc(O)ccc3c(c4ccccc4C(O)=O)c2cc1', 'Oc(c1)ccc2c1oc(cc(O)cc3)c3c42c5c(C(O4)=O)cccc5', 'Oc(c1)ccc2c1oc(cc(=O)cc3)c3c2c4c(C(O)=O)cccc4', 'c1ccc2c(c1)C(=O)OC23c4ccc(cc4Oc5c3ccc(c5)O)O', 'O=C(OC12C3=CC=C(O)C=C3OC4=C2C=CC(O)=C4)C5=C1C=CC=C5', 'O=C(O)c1ccccc1C2=C3C=CC(C=C3Oc4cc(O)ccc42)=O', 'OC(C1=C(C2=C3C=CC(C=C3OC4=C2C=CC(O)=C4)=O)C=CC=C1)=O'],
    'Rhodamine': ['Nc1cc2oc3cc(N)ccc3c(c4ccccc4C(O)=O)c2cc1', 'Nc(c1)ccc2c1oc(cc(=[NH2+])cc3)c3c2c4c(C([O-])=O)cccc4', 'Nc(c1)ccc2c1oc(cc(=[NH2+])cc3)c3c2c4c(C(O)=O)cccc4', 'O=C1c2ccccc2C3(c4c(Oc5cc(N)ccc53)cc(N)cc4)O1', 'O=C1OC2(C3=C(OC4=C2C=CC(N)=C4)C=C(N)C=C3)C5=C1C=CC=C5', 'O=C([O-])c1ccccc1C2=C3C=CC(C=C3Oc4cc(N)ccc42)=[NH2+]', 'O=C(c1c(C2=C3C=CC(C=C3Oc4c2ccc(N)c4)=[NH2+])cccc1)O', '[O-]C(C1=C(C2=C3C=CC(C=C3OC4=C2C=CC(N)=C4)=[NH2+])C=CC=C1)=O'],
    'Rosamine': ['Nc1cc2oc3cc(N)ccc3c(c4ccccc4)c2cc1', 'Nc1cc2c(C(c3ccccc3)=C4C=CC(C=C4O2)=[NH2+])cc1', '[NH2+]=C(C=C1OC2=C3C=CC(N)=C2)C=CC1=C3C4=CC=CC=C4'],
    'Cyanine3': ['CC1(C(/C=C/C=C2Nc3c(C/2(C)C)cccc3)=[NH+]c4c1cccc4)C', 'CC1(C)C2=C([N+]=C1/C=C/C=C3C(C)(C)C4=C(C=CC=C4)N\\3)C=CC=C2'],
    'Cyanine5': ['CC1(C(/C=C/C=C/C=C2Nc3c(C/2(C)C)cccc3)=[NH+]c4c1cccc4)C', 'CC1(C)C2=C([NH+]=C1/C=C/C=C/C=C3C(C)(C)C4=C(C=CC=C4)N\\3)C=CC=C2'],
    'Cyanine7': ['CC1(C(/C=C/C=C/C=C/C=C2Nc3c(C/2(C)C)cccc3)=[NH+]c4c1cccc4)C', 'CC1(C)C2=C([NH+]=C1/C=C/C=C/C=C/C=C3C(C)(C)C4=C(C=CC=C4)N\\3)C=CC=C2'],
    'Pyrene': ['c1cc2cccc3c2c4c1cccc4cc3', 'C12=CC=C3C=CC=C4C=CC(C2=C34)=CC=C1'],
    'Anthracene': ['c1ccc2cc3ccccc3cc2c1', 'C12=CC=CC=C1C=C3C=CC=CC3=C2'],
    'Fluorenone': ['O=C1c2ccccc2c3c1cccc3', 'O=C(C1=CC=CC=C12)C3=C2C=CC=C3'],
    'Xanthone': ['O=C(c1c(O2)cccc1)c3c2cccc3', 'O=C1C2=C(OC3=C1C=CC=C3)C=CC=C2'],
    'Phenoxazine': ['c1(nc(cccc2)c2o3)c3cccc1', 'c1(Nc2c(O3)cccc2)c3cccc1', 'C1(NC2=C(C=CC=C2)O3)=C3C=CC=C1', 'O=C(C=C1O2)C=CC1=Nc3c2cccc3', 'O=C1C=C2OC3=C(C=CC=C3)N=C2C=C1', 'N=C(C=C1O2)C=CC1=Nc3c2cccc3', 'N=C1C=C2OC3=C(C=CC=C3)N=C2C=C1'],
    'BODIPY': ['[B-]1(*)(*)(n2cccc2C=C3C=CC=[N+]31)', '[B-]1(*)(*)(N2C=CC=C2C=C3[N+]1=CC=C3)'],
    'Aza-BODIPY': ['[B-]1(*)(*)n2c(N=C3C=CC=[N+]13)ccc2', '[B-]1(*)(*)(n2cccc2N=C3C=CC=[N+]31)', '[B-]1(*)(*)(N2C=CC=C2N=C3[N+]1=CC=C3)'],
    'Naphthalimide': ['O=C(c1c(c2ccc3)c3ccc1)NC2=O', 'O=C1C2=C3C(C=CC=C3C(N1)=O)=CC=C2'],
}

# Generate RDKit Mol objects for each pattern
patterns_smarts = dict()
for class_name, smiles_list in fluorophore_classes.items():
    pattern_mols = []
    for pattern_smiles in smiles_list:
        pattern_mols.append(Chem.MolFromSmarts(pattern_smiles))
    patterns_smarts[class_name] = pattern_mols

patterns_smiles = dict()
for class_name, smiles_list in fluorophore_classes.items():
    pattern_mols = []
    for pattern_smiles in smiles_list:
        pattern_mols.append(Chem.MolFromSmiles(pattern_smiles))
    patterns_smiles[class_name] = pattern_mols


# Function to classify a molecule based on pattern matching
def classify_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Invalid SMILES'
    matches = []
    for class_name, _ in fluorophore_classes.items():
        for pattern_mol in patterns_smarts[class_name]:
            if mol.HasSubstructMatch(pattern_mol):
                matches.append(class_name)
                break
        
        if not matches:
            for pattern_mol in patterns_smiles[class_name]:
                if pattern_mol is not None:
                    if mol.HasSubstructMatch(pattern_mol):
                        matches.append(class_name)
                        break

    if matches:
        # If a molecule matches multiple classes, join them with commas
        return ', '.join(matches)
    else:
        return 'Other'

# Apply the classification function to the DataFrame
df['class'] = df['smiles'].apply(classify_molecule)

# Save the updated DataFrame to a new CSV file
# The new file will contain the original data plus the 'Class' column
df.to_csv('nchromo_ems_with_classes.csv', index=False)

# Optionally, print out the DataFrame to see the classification results
print(df)