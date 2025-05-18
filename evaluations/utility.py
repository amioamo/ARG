from __future__ import absolute_import, division, print_function

# stdlib
import argparse
import os
from typing import Dict
import numpy as np

import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

np.seterr(divide='ignore', invalid='ignore')

import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')

import sys
sys.path.append('/Users/xw273/Desktop/FMSTY/code/DOMIAS/src/')


# third party
import pandas as pd
import torch
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from utils import *

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.models.ctgan import CTGAN


parser = argparse.ArgumentParser()

parser.add_argument(
    "--snp_nums",
    type = int,
    nargs="+",
    default= [200, 500, 1000],
    help = "sequence length"
)
parser.add_argument(
    "--train_ratios",
    nargs="+",
    default=[0.25, 0.5, 0.75],
    help="training set raitos"
)
parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=[0, 42, 50, 100, 245]
)
parser.add_argument(
    "--train_epoch",
    type=int,
    default=2000,
    help="# training epochs",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='human1000'
)

parser.add_argument(
    "--syn_size",
    type = int,
default = 5000
)

parser.add_argument(
    "--model",
    type = str,
default = 'ctgan'
)



args = parser.parse_args()
dataset = args.dataset
EPOCHS = args.train_epoch
SYN_SIZE = args.syn_size

os.makedirs(f"../../results_folder/{dataset}/{args.model}", exist_ok=True)
print('Now evaluate utility for dataset ' + dataset)
data_file ="../../data/" + dataset + "/real/"
syn_data_file = "../../data/" + dataset + "/syn/" + args.model

utility_logger = {}
for snp_num in args.snp_nums:
    for train_raito in args.train_ratios:

        utility_logger[
            f"{snp_num}_{train_raito}"
        ] = {}
        for i_iter, seed in enumerate(args.seeds):
            ### Load all the dataset
            train = np.load(
                f"../../data/{dataset}/real/{snp_num}/{train_raito}/train_{seed}.npy"
            )
            test = np.load(
                f"../../data/{dataset}/real/{snp_num}/{train_raito}/test_{seed}.npy"
            )
            if args.model == 'ctgan':

                syn = np.load(
                   f"../../data/{dataset}/syn/{args.model}/{snp_num}/{train_raito}/sample_data_epoch_{EPOCHS}_{seed}.pkl",
                        allow_pickle = True
                    ).values

                syn[np.where(syn < 0)[0], np.where(syn < 0)[1]] = 0  # remove the negative values
            else:
                syn = np.load(
                    f"../../data/{args.dataset}/syn/{args.model}/{snp_num}/{train_raito}/sample_data_epoch_{1000}_{seed}.npy",
                    allow_pickle=True
                )
            # though do not know why they exist
            groundtruth = pd.read_csv(f"../../results_folder/{dataset}/{snp_num}_gts.csv")

            """ EVA1. maf """
            maf_ori, maf_ori_g1, maf_ori_g2 = calculate_maf(train[:,:-1], groundtruth)
            maf_val,maf_val_g1, maf_val_g2 = calculate_maf(test[:,:-1], groundtruth)
            maf_syn, maf_syn_g1, maf_syn_g2 = calculate_maf(syn[:,:-1], groundtruth)
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_maf"] = [maf_ori, maf_val, maf_syn]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_maf_g1"] = [maf_ori_g1, maf_val_g1, maf_syn_g1]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_maf_g2"] = [maf_ori_g2, maf_val_g2, maf_syn_g2]

            """ EVA2. heterogensity """
            hetero_ori = calculate_heterozygosity(train[:,:-1])
            hetero_val = calculate_heterozygosity(test[:,:-1])
            hetero_syn = calculate_heterozygosity(syn[:,:-1])
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hetero"] = [hetero_ori, hetero_val, hetero_syn]

            """ EVA3. HWE """
            hwe_ori_chi, hwe_ori_g1_chi, hwe_ori_g2_chi, \
                    hwe_ori_p, hwe_ori_g1_p, hwe_ori_g2_p \
                        = hwe_test(train[:,:-1], groundtruth)

            hwe_val_chi, hwe_val_g1_chi, hwe_val_g2_chi, \
                hwe_val_p, hwe_val_g1_p, hwe_val_g2_p \
                = hwe_test(test[:,:-1], groundtruth)

            hwe_syn_chi, hwe_syn_g1_chi, hwe_syn_g2_chi, \
            hwe_syn_p, hwe_syn_g1_p, hwe_syn_g2_p\
                = hwe_test(syn[:,:-1], groundtruth)

            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_chi"] = [hwe_ori_chi, hwe_val_chi, hwe_syn_chi]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_chi_g1"] = [hwe_ori_g1_chi, hwe_val_g1_chi, hwe_syn_g1_chi]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_chi_g2"] = [hwe_ori_g2_chi, hwe_val_g2_chi, hwe_syn_g2_chi]

            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_p"] = [hwe_ori_p, hwe_val_p, hwe_syn_p]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_p_g1"] = [hwe_ori_g1_p, hwe_val_g1_p, hwe_syn_g1_p]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_hwe_p_g2"] = [hwe_ori_g2_p, hwe_val_g2_p, hwe_syn_g2_p]

            """ EVA4. missing data imputation """
            imputer = KNNImputer(n_neighbors = 3)
            #rmse_s, rmse_r,  = imputation(0.1, 0.05, train, syn, test, imputer)

            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_imputation_0.1"] = imputation(0.1, 0.05, train[:,:-1], syn[:,:-1], \
                                                 test[:,:-1], imputer, [0, 42, 50, 100, 245])

            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_imputation_0.2"] = imputation(0.2, 0.05, train[:, :-1], syn[:, :-1], \
                                                 test[:, :-1], imputer, [0, 42, 50, 100, 245])

            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_imputation_0.3"] = imputation(0.3, 0.05, train[:, :-1], syn[:, :-1], \
                                                 test[:, :-1], imputer, [0, 42, 50, 100, 245])



            """ EVA5. GWAS results """
            mae_group_1, mae_group_2, mae_whole, \
            accuracy_group_1, accuracy_group_2, accuracy_whole = association_test(syn, groundtruth)
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_gwas_mae"] = [mae_group_1, mae_group_2, mae_whole]
            utility_logger[
                f"{snp_num}_{train_raito}"
            ][f"{seed}_gwas_acc"] = [accuracy_group_1, accuracy_group_2, accuracy_whole]

#print(utility_logger[f"{200}_{0.25}"]['0_gwas_acc'])


with open(f'../../results_folder/{dataset}/{args.model}/utility_logger.pkl', 'wb') as pf:
    pickle.dump(utility_logger, pf)
