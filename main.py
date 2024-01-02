import numpy as np
from scipy import integrate
from scipy.linalg import expm
import nlopt
import argparse
import csv
from math import exp
import json
from methods import method_nd_md, method_nd_ms, method_ns_md, method_ns_ms

# setup args
def setup_args():
    parser = argparse.ArgumentParser(description="exp")
    parser.add_argument('--mode', type=str,
                        default="nd_md", help="mode: nd_md, nd_ms, ns_md, ns_ms")
    parser.add_argument('--data-path', type=str,
                        default="./data/nd_md.csv", help="path to laod data")
    parser.add_argument('--par-path', type=str,
                        default="./par/par_nd_md.json", help="path to load par")
    parser.add_argument('--result-path', type=str,
                        default="./result/paraest_n3_nd_md_n500.txt", help="path to save result")    
    return parser.parse_args()

def load_par(par_path):
    with open(par_path) as f:
        par = json.load(f)
    return par

def load_data(data_path):
    csv_filename = data_path
    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        columns = {col: [] for col in header}
        for row in reader:
            for col in header:
                columns[col].append(float(row[col]))
    for col in header:
        print(f'{col}: {columns[col]}')    
    return columns

def load_method(mode):
    if mode=="nd_md":
        method = method_nd_md
    elif mode=="nd_ms":
        method = method_nd_ms
    elif mode=="ns_md":
        method = method_ns_md
    elif mode=="ns_ms":
        method = method_ns_ms

    return method


if __name__=='__main__':
    args = setup_args()
    #print(args)
    par = load_par(args.par_path) 
    #print(par)

    data_all = load_data(args.data_path)

    method = load_method(args.mode)

    method(args.result_path, par, data_all)

