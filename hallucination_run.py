import subprocess
import torch
from str2bool import str2bool
import esm
import os
import glob
import numpy as np
import biotite.structure.io as bsio
from itertools import islice
import re
import sys
import argparse
from Bio import PDB
import pandas as pd
import random

def extract_atoms_from_model(input_pdb_file, output_pdb_file, target_model_id):
    
    with open(input_pdb_file, "r") as input_file:
        with open(output_pdb_file, "w") as output_file:
            for line in input_file:
                if line.startswith("MODEL") and "1" not in line:
                    break
                if line.startswith("ATOM") and line[21] == target_model_id:
                    # Extract only the necessary information from the ATOM line
                    output_file.write(line)


def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs, key=len)]
 
    return z


def extract_scope():

    matches = ["c.1.", "c.2.", "c.37.", "d.58", "a.4", "c.23", "c.55", "b.40", "c.66"]

    name_dict = { '1' : "TIM-barrel_c.1", '2' : "Rossman-fold_c.2", '37' : "P-fold_Hydrolase_c.37",
                '58' : "Ferredoxin_d.58", '4' : "DNA_RNA-binding_3-helical_a.4", '23' : "Flavodoxin-like_c.23",
                '55' : "Ribonuclease_H-like_motif_c.55", '40' : "OB-fold_greek-key_b.40", '66' : "Nucleoside_Hydrolase_c.66"}

    path = r"/home/sofia/RFDiffusion/RFdiffusion/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"

    with open(path) as f:
        temp_string = []
        temp_seqs = []
        temp_class = []
        temp_sub = []
        temp_name = []
        correct_match = False
        for line in f:
            if correct_match and not line.startswith(">"):
                temp_string.append(line.strip())
                continue
            else:
                if len(temp_string) != 0:
                    temp = ' '.join(temp_string).strip().upper()
                    temp_seqs.append(re.sub(r"[\n\t\s]*", "", temp))
                    temp_string = []
                correct_match = False
                result1 = re.findall('c\.(1|2|23|37|55|66)\.\d*\.\d*', line)
                result2 = re.findall('d\.(58)\.\d*\.\d*', line)
                result3 = re.findall('a\.(4)\.\d*\.\d*', line)
                result4 = re.findall('b\.(40)\.\d*\.\d*', line)
                if len(result1) != 0:
                    temp_class.append(result1[0])
                    correct_match = True
                elif len(result2) != 0:
                    temp_class.append(result2[0])
                    correct_match = True
                elif len(result3) != 0:
                    temp_class.append(result3[0])
                    correct_match = True
                elif len(result4) != 0:
                    temp_class.append(result4[0])
                    correct_match = True
                result1 = re.findall('c\.(1|2|23|37|55|66)\.\d*\.\d*', line)
                result2 = re.findall('d\.58\.\d*\.\d*', line)
                result3 = re.findall('a\.4\.\d*\.\d*', line)
                result4 = re.findall('b\.40\.\d*\.\d*', line)
                if len(result1) != 0:
                    temp_sub.append(re.search('c\.(1|2|23|37|55|66)\.\d*\.\d*', line).group())
                elif len(result2) != 0:
                    temp_sub.append(result2[0])
                elif len(result3) != 0:
                    temp_sub.append(result3[0])
                elif len(result4) != 0:
                    temp_sub.append(result4[0])
                if correct_match and line.startswith(">"):
                    temp_name.append(line[1:8])

    f.close()
        
    seqs = np.array(temp_seqs)
    classes = np.array(temp_class)
    subclass = np.array(temp_sub)
    names = np.array(temp_name)
    
    res = min(temp_seqs, key=len)
    arg = np.argwhere(seqs==res)

    sort_seqs = np.array(sorted(seqs, key=len))
    args = []
    lens = []
    for x in sort_seqs:
        temp = np.argwhere(seqs == x)
        args.append(temp[0][0])
        lens.append(len(x))

    classnames = []
    for x in classes[args]:
        classnames.append(name_dict[x])

    df = pd.DataFrame( {'Length': lens, 'Classes': classes[args].astype(np.int64), 'Seqs': sort_seqs, "Names" : classnames, "Subclass" : subclass[args], "SCOPe name": names[args]} )

    df_short = df[df['Length'] <= 100]

    return df_short[df_short['Length'] >= 50]


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pipeline', type=str, required=False, default="True", help='Whether to run pipeline or not.')
    parser.add_argument('--protein', type=str, required=False, default="6IY4", help='Name of PDB file.')

    args = parser.parse_args()
    
    return args.protein, str2bool(args.pipeline)

protein, pipeline = parse_arguments()

if pipeline:
    df = extract_scope()
    scopepath = "/home/shared/databases/SCOPe/unrelaxed/native/"
    out_prefix = "/mnt/nasdata/sofia/Protein_Evo/"
    
    scope_names = df['SCOPe name'].tolist()
    scope_class = df['Names'].tolist()
    
    babypdb = []
    inputpdb = []
    outscaff = []
    outpath = []
    outdir = []
    for name,cl in zip(scope_names,scope_class):
        babypdb.append(scopepath + name[2:4] + "/" + name + ".ent")
            
        temp_dir = out_prefix + cl
        
        try:
            os.makedirs(temp_dir, exist_ok = True)
            print("Directory '%s' created successfully" % temp_dir)
        except OSError as error:
            print("Directory '%s' can not be created" % temp_dir)
            
        inputpdb.append(out_prefix + cl + "/" + name + ".pdb")
        outscaff.append(out_prefix + cl + "/SStruct/" + name)
        outpath.append(out_prefix + cl + "/" + name + "/test_" + name)
        outdir.append(out_prefix + cl + "/" + name)
else:
    babypdb = ["/home/sofia/RFDiffusion/RFdiffusion/Protein_evo/" + protein + ".pdb"]
    inputpdb = ["/home/sofia/RFDiffusion/RFdiffusion/Protein_evo/" + protein + "edit.pdb"]
    outscaff = ["/home/sofia/RFDiffusion/RFdiffusion/Protein_evo/SStruct/" + protein]
    outpath = ["/home/sofia/RFDiffusion/RFdiffusion/Protein_evo/" + protein + "/test" + protein.lower()]
    outdir = ["/home/sofia/RFDiffusion/RFdiffusion/Protein_evo/" + protein]

for bpdb, ipdb, oscaff, opath, odir in zip (babypdb, inputpdb, outscaff, outpath, outdir):

    # Skip if file already exists
    if os.path.isfile(ipdb):
        continue
    else:
        if os.path.isfile(bpdb):
            extract_atoms_from_model(bpdb, ipdb, "A")
        # Skip if file does not exist in database
        else:
            continue

    command1 = "./helper_scripts/make_secstruc_adj.py --input_pdb " + ipdb + " --out_dir " + oscaff
    command2 = "./scripts/run_inference.py inference.output_prefix=" + opath + " scaffoldguided.scaffoldguided=True scaffoldguided.target_pdb=False scaffoldguided.scaffold_dir=" + oscaff

    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)

    bias_command = "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/make_bias_AA.py --output_path " + odir + "/bias.jsonl \
        --AA_list 'N K Q R C H F M Y W'\
        --bias_list '-10 -10 -10 -10 -10 -10 -10 -10 -10 -10'"

    subprocess.run(bias_command, shell=True)

    json_command = "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path " + odir + " --output_path " + odir + "/test.jsonl"

    subprocess.run(json_command, shell=True)

    command3 = "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/protein_mpnn_run.py --num_seq_per_target=20 --batch_size=10 --out_folder=" + odir + " --jsonl_path=" + odir + "/test.jsonl  --bias_AA_jsonl " + odir + "/bias.jsonl"

    subprocess.run(command3, shell=True)

"""bad_aa = ["N", "K", "Q", "R", "C", "H", "F", "M", "Y", "W"]

path = outdir + '/seqs/'

files = glob.glob(path + "/*.fa")

seq = []
score = []
pname = []

for fname in files:
    filename, ext = os.path.splitext(fname)
    filename = os.path.basename(fname)
    if ext == '.fa':
        pname.append(filename.replace(ext, ''))
        with open(fname) as f:
            temp_seq = []
            temp_score = []
            for line in islice(f, 2, None):
                result = re.findall('global_score=\d*\.?\d*', line)
                if len(result) == 0:
                    temp_seq.append(line.replace('\n', ''))
                else:
                    sc = result[0].replace('global_score=', '')
                    temp_score.append(float(sc))
        seq.append(np.array(temp_seq, dtype=str))
        score.append(np.array(temp_score))
        f.close()

seq = np.array(seq)
score = np.array(score)
pname = np.array(pname)

bool_arr = []

for ss in seq:
    temp_bool = []
    for s in ss: 
        arr = [1 for e in bad_aa if e in s]
        if len(arr) == 0:
            temp_bool.append(True)
        else:
            temp_bool.append(False)
    bool_arr.append(np.array(temp_bool, dtype=bool))

bool_arr = np.array(bool_arr)

print(np.all(bool_arr == True))

minargs = np.argmin(score, axis=1)

seqs = []

for i,j in enumerate(minargs):
    seqs.append(seq[i,j])"""

"""
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
model.set_chunk_size(128)

for sequence,name in zip(seqs,pname):
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(path+name+".pdb", "w") as f:
        f.write(output)

    struct = bsio.load_structure(path+name+".pdb", extra_fields=["b_factor"])
    print(name)
    print(struct.b_factor.mean())  # this will be the pLDDT
"""