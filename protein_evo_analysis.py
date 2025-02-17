import subprocess
import torch
import esm
import argparse
import os
import glob
import numpy as np
import biotite.structure.io as bsio
from itertools import islice
import re
import sys
import random
import concurrent.futures
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
import Levenshtein as lev
import itertools
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Selection, DSSP
from Bio.PDB.Polypeptide import one_to_three
from Bio.SVDSuperimposer import SVDSuperimposer
from str2bool import str2bool
import json
from timeit import default_timer as timer
from pathlib import Path
from pynvml import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import textwrap


def wrap_labels(ax, width, break_long_words=True):
    """
    Wraps the x-axis labels of a given matplotlib Axes object to a specified width.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object containing the x-axis labels to be wrapped.
    width (int): The maximum line width for the wrapped labels.
    break_long_words (bool, optional): Whether to break long words. Defaults to True.

    Returns:
    None
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


AA = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def superpose_structures(pdb_file1, pdb_file2):
    # Full backbone alignment
    atom_types = ["CA", "N", "C", "O"]

    parser = Bio.PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("protein1", pdb_file1)
    structure2 = parser.get_structure("protein2", pdb_file2)

    coords1 = [
        a.coord
        for a in structure1[0].get_atoms()
        if a.parent.resname in AA and a.name in atom_types
    ]
    coords2 = [
        a.coord
        for a in structure2[0].get_atoms()
        if a.parent.resname in AA and a.name in atom_types
    ]

    if len(coords1) != len(coords2):
        return 10

    # Now we initiate the superimposer:
    super_imposer = SVDSuperimposer()
    super_imposer.set(np.array(coords1), np.array(coords2))
    super_imposer.run()

    return super_imposer.get_rms()


def calculate_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def flatten_chain(matrix):
    return list(itertools.chain.from_iterable(matrix))


def calculate_lddt_score(protein1_pdb_path, protein2_pdb_path, inclusion_radius=15.0):
    thresholds = [0.5, 1.0, 2.0, 4.0]
    num_residues = 0
    num_residues_within_thresholds = [0] * len(thresholds)

    parser = Bio.PDB.PDBParser(QUIET=True)
    protein1 = parser.get_structure("protein1", protein1_pdb_path)
    protein2 = parser.get_structure("protein2", protein2_pdb_path)

    if len(protein1) != len(protein2):
        raise ValueError("The two proteins must have the same number of chains.")

    for chain1, chain2 in zip(protein1.get_chains(), protein2.get_chains()):
        if len(chain1) != len(chain2):
            raise ValueError("The two chains must have the same number of residues.")

        for res1, res2 in zip(chain1, chain2):
            # if len(res1) != len(res2):
            # raise ValueError("The two residues must have the same number of atoms.")

            ca_atoms1 = [atom1 for atom1 in res1 if atom1.get_name() == "CA"]
            ca_coord1 = ca_atoms1[0].get_coord() if ca_atoms1 else None

            if ca_coord1 is None:
                continue

            num_residues += 1

            distances = [
                calculate_distance(ca_coord1, atom2.get_coord())
                for atom2 in res2
                if atom2.get_name() != "CA"
            ]
            for i, threshold in enumerate(thresholds):
                if any(
                    distance <= (inclusion_radius + threshold) for distance in distances
                ):
                    num_residues_within_thresholds[i] += 1

    fractions_within_thresholds = [
        num_res / num_residues for num_res in num_residues_within_thresholds
    ]
    lddt_score = sum(fractions_within_thresholds) / len(fractions_within_thresholds)

    return lddt_score


# Define a function to shorten consecutive letters
def shorten_secondary_structure(ss):
    shortened_ss = ""
    prev_char = ""
    for char in ss:
        if char != prev_char:
            shortened_ss += char
        prev_char = char
    return shortened_ss


# Process each PDB file
def get_dssp(pdb_file, simplify=False):
    # Load an existing PDB file
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)

    # Create a custom header line
    header = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"
    new_file = pdb_file[:-4] + "_edit.pdb"

    # Open the PDB file for writing
    with open(new_file, "w") as file:
        # Write the custom header line
        file.write(header)

        # Create a PDBIO instance to write the structure
        pdb_io = PDBIO()

        # Write the rest of the structure (atoms, coordinates, etc.)
        pdb_io.set_structure(structure)
        pdb_io.save(file)

    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", new_file)

    # Calculate DSSP information
    model = structure[0]
    dssp = DSSP(model, new_file)

    # Extract the secondary structure string
    secondary_structure = "".join([dssp[key][2] for key in dssp.keys()])

    simplified_mapping = {
        "H": "H",  # Helix
        "G": "H",  # 3-10 helix
        "I": "H",  # Pi helix
        "P": "H",  # Poly-proline II helix (Treat as Helix)
        "E": "E",  # Extended strand (Beta sheet/strand)
        "B": "E",  # Isolated beta-bridge residue
        "T": "C",  # Turn
        "S": "C",  # Bend
        "C": "C",  # Coil or unassigned region
        "-": "C",  # Treat "-" as Coil (unassigned or gap)
    }

    # Simplify to three classifications
    simplified_string = "".join(
        simplified_mapping[symbol] for symbol in secondary_structure
    )

    # Shorten consecutive letters
    shortened_ss = shorten_secondary_structure(simplified_string)

    # Delete temporary DSSP pdb file
    os.remove(new_file)

    if simplify:
        return shortened_ss.replace("C", "")
    else:
        return simplified_string


def choose_best_seq(path):
    seqs, pname, score = get_best_seqs(path)

    best_i = np.argmin(score)
    best_seq = seqs[best_i]
    best_name = pname[best_i]

    return best_seq, best_name.replace(".fa", ".pdb")


def get_best_seqs(path):
    if path.endswith(".pdb"):
        temp = path.split("/")
        path3 = ""
        for i in range(len(temp) - 1):
            path3 += temp[i] + "/"
        path3 += "seqs/"
        path3 += temp[-1].replace(".pdb", ".fa")
        files = [path3]
    else:
        files = glob.glob(path + "/*.fa")

    seq = []
    score = []
    pname = []

    for fname in files:
        filename, ext = os.path.splitext(fname)
        filename = os.path.basename(fname)
        if ext == ".fa":
            pname.append(filename.replace(ext, ""))
            with open(fname) as f:
                temp_seq = []
                temp_score = []
                for line in islice(f, 2, None):
                    result = re.findall("global_score=\d*\.?\d*", line)
                    if len(result) == 0:
                        temp_seq.append(line.replace("\n", ""))
                    else:
                        sc = result[0].replace("global_score=", "")
                        temp_score.append(float(sc))
            seq.append(np.array(temp_seq, dtype=str))
            score.append(np.array(temp_score))
            f.close()

    seq = np.array(seq)
    score = np.array(score)
    pname = np.array(pname)

    minargs = np.argmin(score, axis=1)

    seqs = []

    for i, j in enumerate(minargs):
        seqs.append(seq[i, j])

    return seq, pname, score


def esm_structs(path):
    print("IN ESM STRUCTURE DETERMINATION")

    seq, pname, score = get_best_seqs(path)

    seqs = []

    minargs = np.argmin(score, axis=1)

    for i, j in enumerate(minargs):
        seqs.append(seq[i, j])

    model = esm.pretrained.esmfold_v1()
    CUDA_PREFIX = "cuda:"
    if gpu_name == 0 or gpu_name == 1:
        device_name = CUDA_PREFIX + str(gpu_name)
    else:
        available_mems = np.array(
            [get_mem_device(gpu) for gpu in range(torch.cuda.device_count())]
        )
        device_name = CUDA_PREFIX + str(np.argmin(available_mems))

    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    model.set_chunk_size(128)

    plddt = {}

    for sequence, name in zip(seqs, pname):
        with torch.no_grad():
            output = model.infer_pdb(sequence)

        with open(path[:-5] + name + "ESMfold.pdb", "w") as f:
            f.write(output)

        struct = bsio.load_structure(
            path[:-5] + name + "ESMfold.pdb", extra_fields=["b_factor"]
        )
        print(name)
        print(struct.b_factor.mean())  # this will be the pLDDT

        dssp_base = get_dssp(str(os.path.split(path)[0]) + ".pdb", simplify=True)
        new_dssp = get_dssp(path[:-5] + name + "ESMfold.pdb", simplify=True)

        if dssp_base.count("H") == new_dssp.count("H") and dssp_base.count(
            "E"
        ) == new_dssp.count("E"):
            plddt["DSSP"] = "True"
        else:
            plddt["DSSP"] = "False"

        plddt["Path"] = path
        plddt["Class"] = path.split("/")[-3]
        plddt["Sequence"] = sequence
        plddt["pLDDT"] = struct.b_factor.mean()
        plddt["ESM path"] = path[:-5] + name + "ESMfold.pdb"

    return plddt


def esm_generation(path):
    print("IN ESM STRUCTURE DETERMINATION")

    seq, pname, score = get_best_seqs(path)

    seqs = []

    minargs = np.argmin(score, axis=1)

    for i, j in enumerate(minargs):
        seqs.append(seq[i, j])

    model = esm.pretrained.esmfold_v1()
    if gpu_name == 0 or 1:
        device_name = "cuda:" + str(gpu_name)
    else:
        available_mems = np.array(
            [get_mem_device(gpu) for gpu in range(torch.cuda.device_count())]
        )
        device_name = "cuda:" + str(np.argmin(available_mems))
    model = model.eval().to(device_name)

    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    model.set_chunk_size(64)

    paths = []

    for sequence, name in zip(seqs, pname):
        with torch.no_grad():
            output = model.infer_pdb(sequence)

        with open(path[:-5] + name + "ESMfold.pdb", "w") as f:
            f.write(output)

        paths.append(path[:-5] + name + "ESMfold.pdb")

    return paths


sns.set_theme()

gpu_name = 0

name_dict = {
    "1": "TIM-barrel_c.1",
    "2": "Rossman-fold_c.2",
    "37": "P-fold_Hydrolase_c.37",
    "58": "Ferredoxin_d.58",
    "4": "DNA_RNA-binding_3-helical_a.4",
    "23": "Flavodoxin-like_c.23",
    "55": "Ribonuclease_H-like_motif_c.55",
    "40": "OB-fold_greek-key_b.40",
    "66": "Nucleoside_Hydrolase_c.66",
}

fold_types = list(name_dict.values())

path = "/mnt/nasdata/sofia/Protein_Evo/"

new_files = []

print("Checking if csv file exists.")
if not os.path.exists("protein_evo_results.csv"):
    print("File does not exist. Analysing data.")

    for fold in fold_types:
        pathdir = path + fold

        if os.path.exists(pathdir):
            files = glob.glob(pathdir + "/*.pdb")
            file_exist = []
            for f in files:
                if os.path.exists(pathdir + "/" + os.path.basename(f)[:-4]):
                    file_exist.append(f)

            print(
                f"For the fold {fold}, there were {len(file_exist)} files generated. {len(files) - len(file_exist)} files were invalid."
            )

            new_files.append(file_exist)

    new_files = flatten_chain(new_files)

    lddt_files = dict()
    lddt_scores = []
    rmsds = []
    dssp_bool = []
    classes = []
    accept_arr = []
    filepaths = []
    og_dssp = []
    calc_dssp = []

    for template, i in zip(new_files, tqdm(range(len(new_files)))):
        pathdir = template[:-4]
        pdbfiles = glob.glob(pathdir + "/*.pdb")

        dssp_base = get_dssp(template, simplify=True)

        accepted = []

        for newpdb in pdbfiles:
            temp_score = calculate_lddt_score(template, newpdb)
            lddt_scores.append(temp_score)
            rmsds.append(superpose_structures(template, newpdb))

            # Check if secondary structure is correct
            new_dssp = get_dssp(newpdb, simplify=True)
            og_dssp.append(dssp_base)
            calc_dssp.append(new_dssp)

            if dssp_base.count("H") == new_dssp.count("H") and dssp_base.count(
                "E"
            ) == new_dssp.count("E"):
                if rmsds[-1] <= 5:
                    accepted.append(newpdb)
                    accept_arr.append("True")
                else:
                    accept_arr.append("False")
                dssp_bool.append("True")
            else:
                dssp_bool.append("False")
                accept_arr.append("False")

            namelist = newpdb.split("/")
            classes.append(namelist[-3])
            filepaths.append(newpdb)

        if len(accepted) > 0:
            lddt_files[template] = accepted

    print(lddt_files)
    print(len(flatten_chain(list(lddt_files.values()))))
    print(len(list(lddt_files.keys())))

    lddt_scores = np.array(lddt_scores)
    rmsds = np.array(rmsds)
    dssp_bool = np.array(dssp_bool)
    classes = np.array(classes)
    accept_arr = np.array(accept_arr)
    filepaths = np.array(filepaths)
    og_dssp = np.array(og_dssp)
    calc_dssp = np.array(calc_dssp)

    df = pd.DataFrame(
        {
            "LDDT": lddt_scores,
            "RMSD": rmsds,
            "DSSP": dssp_bool,
            "Fold": classes,
            "Accepted": accept_arr,
            "Paths": filepaths,
            "Target DSSP": og_dssp,
            "New DSSP": calc_dssp,
        }
    )

    df.to_csv("protein_evo_results.csv", index=False)
else:
    print("File exists. Reading in data.")
    df = pd.read_csv("protein_evo_results.csv")
    print("Finished reading in data.")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
sns.histplot(data=df, x="LDDT", hue="Accepted", kde=True, ax=ax1)
sns.histplot(data=df, x="RMSD", hue="Accepted", kde=True, ax=ax2)
sns.histplot(data=df, x="RMSD", hue="DSSP", kde=True, ax=ax3)
ax1.set_xlabel("LDDT scores")
ax1.text(
    0.5,
    2500,
    f"{sum((df['LDDT'].to_numpy() >= 0.8))} accepted on LDDT",
    bbox=dict(facecolor="red", alpha=0.5),
)
ax2.set_xlabel("RMSD")
ax2.text(
    20,
    250,
    f"{sum((df['RMSD'].to_numpy() <= 5))} accepted on RMSD",
    bbox=dict(facecolor="blue", alpha=0.5),
)
ax3.set_xlabel("RMSD")
ax3.text(
    20,
    200,
    f"{sum((df['DSSP'].to_numpy() == True).astype(int))} accepted on DSSP",
    bbox=dict(facecolor="green", alpha=0.5),
)
plt.savefig("LDDT_score_dist_hue.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
sns.histplot(data=df, x="Fold", hue="Accepted", kde=False, ax=ax[0, 0])
sns.histplot(data=df, x="Fold", hue="DSSP", kde=False, ax=ax[0, 1])
sns.histplot(data=df, x="LDDT", hue="Fold", kde=True, ax=ax[1, 0])
sns.histplot(data=df, x="RMSD", hue="Fold", kde=True, ax=ax[1, 1])
wrap_labels(ax[0, 0], 12)
wrap_labels(ax[0, 1], 12)
plt.savefig("class_distributions.pdf", bbox_inches="tight")
plt.show()

sns.displot(data=df, x="RMSD", hue="Fold", col="DSSP")
plt.savefig("RMSD_fold_distribution.pdf", bbox_inches="tight")
plt.show()

f, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=df, x="RMSD", y="LDDT", hue="Fold", ax=axs[0])
sns.histplot(
    data=df, x="Fold", hue="Fold", shrink=0.8, alpha=0.8, legend=False, ax=axs[1]
)
wrap_labels(axs[1], 12)
f.tight_layout()
plt.savefig("RMSD_vs_LDDT_counts.pdf", bbox_inches="tight")
plt.show()

sns.displot(data=df, x="DSSP", col="Fold", facet_kws={"sharey": False})
plt.savefig("accepted_folds.pdf", bbox_inches="tight")
plt.show()

df_accept = df[df["DSSP"] == "True"]
dssp_paths = df_accept["Paths"].to_numpy()

if not os.path.exists("protein_structure_evaluation_dssp.csv"):
    print("Performing initial structure validation.")
    df_struct_list = []

    for path, i in zip(dssp_paths, tqdm(range(len(dssp_paths)))):
        temp_dict = esm_structs(path)
        df_struct_list.append(temp_dict)

    df_struct = pd.DataFrame(df_struct_list)
    df_struct.to_csv("protein_structure_evaluation_dssp.csv", index=False)

else:
    print("Reading in structure valuation data.")
    df_struct = pd.read_csv("protein_structure_evaluation_dssp.csv")


sns.displot(
    data=df_struct, x="pLDDT", col="Class", row="DSSP", facet_kws={"sharey": False}
)
plt.savefig("pLDDT_validation_class.pdf", bbox_inches="tight")
plt.show()
