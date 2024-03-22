import os
import re

import numpy as np
import pandas as pd

# from rdkit.Chem.rdmolfiles import MolFromSDFFile
from rdkit.Chem import Descriptors
from rdkit import Chem

from tqdm import tqdm

if __name__ == "__main__":
    fail_count = 0
    file_names = os.listdir("dipole_sousa")
    all_smiles = []
    all_dipole = []
    dipole = pd.read_csv('dipole_moments_10071mols.csv')
    for f in tqdm(file_names):
        # get the smiles
        try:
            mol = Chem.SDMolSupplier(
                os.path.join("dipole_sousa", f)
                # sanitize=False,
                # proximityBonding=True,
                # removeHs=True,
            )
        # mol = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(next(mol))
            all_smiles.append(smiles)
            all_dipole.append(dipole[dipole['Molecule_ID']==f]['Dipole_Moment(D)'].item())
                
  
            # get the dipole
            # with open(
            #     os.path.join("ornl_aisd_data", f, "detailed.out"), "r"
            # ) as details:
            #     # get the third-to-last line
            #     lines = details.readlines()
            #     line = lines[-3]

            #     pattern = r"[-+]?\d*\.\d+|\d+"

            #     # use re.findall to extract numerical values
            #     numerical_values = re.findall(pattern, line)

            #     # convert the strings to floats
            #     dipole = np.array([float(value) for value in numerical_values])
            #     all_dipole.append(np.sqrt(np.sum(dipole**2)))

        except:
            fail_count += 1

    print(
        "successfully loaded {0} out of {1} smiles.pdb".format(
            (len(file_names) - fail_count), len(file_names)
        )
    )
    # get features
    features = [
        "ExactMolWt",
        "TPSA",
        "MolLogP",
        "NumValenceElectrons",
        "NumRadicalElectrons",
        "MaxPartialCharge",
        "NHOHCount",
        "NOCount",
        "NumAromaticRings",
        "NumHAcceptors",
        "NumHDonors",
        "NumHeteroatoms",
        "NumRotatableBonds",
        "RingCount",
        "MolMR",
        "NumAliphaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumAliphaticRings",
        "NumAromaticCarbocycles",
        "NumAromaticHeterocycles",
        "NumSaturatedCarbocycles",
        "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "HeavyAtomMolWt",
    ]

    descrs = [
        [
            getattr(Descriptors, feature)(Chem.MolFromSmiles(smile))
            for feature in features
        ]
        for smile in all_smiles
    ]

    # convert to a dataframe
    df = pd.DataFrame(descrs, columns=features)
    df["smiles"] = all_smiles
    df["dipole"] = all_dipole
    print(df)
    df.to_csv("sousa_molecules.csv")
