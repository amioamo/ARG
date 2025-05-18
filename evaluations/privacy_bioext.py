#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
import umap

# Append DOMIAS source path if needed

# Global definitions for density estimation methods
def compute_freq_likelihood(train_data):
    """
    Given a numpy array train_data (N x D), computes genotype frequencies for each column.
    Returns a function that, given a sample x (of shape (D,)), computes the product of frequencies.
    """
    N, D = train_data.shape
    freq = [{} for _ in range(D)]
    for j in range(D):
        unique, counts = np.unique(train_data[:, j], return_counts=True)
        for u, c in zip(unique, counts):
            freq[j][u] = c / N

    def p_est(x):
        p = 1.0
        for j, val in enumerate(x):
            p *= freq[j].get(val, 1e-9)
        return p
    return p_est


def discrete_kde(train_data, alpha=0.5):
    """
    A discrete KDE using a Hamming-based kernel on train_data.
    Returns a function that, for a given sample x, computes the average kernel value.
    """
    train_data = np.array(train_data)
    N, D = train_data.shape
    def p_est(x):
        # Compute Hamming distance between x and all rows of train_data
        dist = np.sum(train_data != x, axis=1)
        k_val = np.exp(-alpha * dist)
        return k_val.mean()
    return p_est

# Import DOMIAS modules
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.models.ctgan import CTGAN

# -------------------------------
# New parser arguments
parser = argparse.ArgumentParser(description="Extended Privacy Evaluation Framework")
parser.add_argument("--density_estimator", type=str, default="kde",
                    choices=["bnaf", "kde", "freq", "discrete_kde"],
                    help="Which approach to use for computing p_G(x) and p_R(x).")
parser.add_argument("--embedding_method", type=str, default="pca", choices=["pca", "umap"],
                    help="Which embedding method to use for density estimation and CTGAN transformer embedding.")
parser.add_argument("--snp_nums", nargs="+", type=int, default=[200, 500, 1000],
                    help="SNP sequence lengths.")
parser.add_argument("--train_ratios", nargs="+", default=[0.25, 0.5, 0.75],
                    help="Training set ratios.")
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 50, 100, 245],
                    help="Random seeds to use.")
parser.add_argument("--dataset", type=str, default='1kgp',
                    help="Dataset name.")
parser.add_argument("--train_epoch", type=int, default=2000,
                    help="Number of training epochs for synthetic generation.")
parser.add_argument("--model", type=str, default='ctgan',
                    help="Model name used for synthetic data generation.")
parser.add_argument("--rep_dim", type=int, default=128,
                    help="Latent representation dimension.")
parser.add_argument("--gpu_idx", default=None, type=int,
                    help="GPU index to use, if any.")
args = parser.parse_args()

# Set device
if args.gpu_idx is not None and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_idx)

if torch.cuda.is_available():
    device = torch.device("cuda")

# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

LATENT_REPRESENTATION_DIM = args.rep_dim
os.makedirs(f"../results_folder/{args.dataset}/{args.model}", exist_ok=True)
print("Now evaluating privacy for dataset " + args.dataset)

# Define file paths for real and synthetic data
data_file = os.path.join("..", "..", "data", args.dataset, "real")
syn_data_file = os.path.join("..", "..", "data", args.dataset, "syn", args.model)

privacy_logger: Dict = {}

# Loop over SNP lengths and training ratios
for snp_num in args.snp_nums:
    for train_ratio in args.train_ratios:
        key_base = f"{args.density_estimator}_{args.embedding_method}_{snp_num}_{train_ratio}"
        privacy_logger[key_base] = {}
        for seed in args.seeds:
            # Load real data
            train_path = os.path.join(data_file, str(snp_num), str(train_ratio), f"train_{seed}.npy")
            test_path = os.path.join(data_file, str(snp_num), str(train_ratio), f"test_{seed}.npy")
            ref_path = os.path.join(data_file, str(snp_num), "ref.npy")  # reference is fixed

            train = np.load(train_path)
            test = np.load(test_path)
            ref = np.load(ref_path)
            X_test = np.concatenate([train, test])
            Y_test = np.concatenate([np.ones(train.shape[0]), np.zeros(test.shape[0])]).astype(bool)

            # Load synthetic data
            if args.model == 'ctgan':
                syn_path = os.path.join(syn_data_file, str(snp_num), str(train_ratio),
                                        f"sample_data_epoch_{args.train_epoch}_{seed}.pkl")
                syn_loaded = np.load(syn_path, allow_pickle=True)
                if isinstance(syn_loaded, dict):
                    syn = list(syn_loaded.values())
                elif isinstance(syn_loaded, pd.DataFrame):
                    syn = syn_loaded.values
                else:
                    syn = syn_loaded
                # Remove negative values if any
                neg_idx = np.where(syn < 0)
                if neg_idx[0].size > 0:
                    syn[neg_idx] = 0
            else:
                syn_path = os.path.join(syn_data_file, str(snp_num), str(train_ratio),
                                        f"sample_data_epoch_{1000}_{seed}.npy")
                syn = np.load(syn_path, allow_pickle=True)



            # Split synthetic data into samples and validation
            num_test = len(X_test)
            samples = syn[: num_test // 2]
            samples_val = syn[num_test // 2:]

            # -------------------------------
            # Embedding / Dimensionality Reduction Block
            # Using the selected embedding_method for density estimators bnaf, kde, and umap.
            if args.density_estimator in ["bnaf", "kde"]:
                if args.embedding_method == "pca":
                    print(f"Seed {seed}: Using PCA for embedding...")
                    pca = PCA(n_components=64)
                    samples_embed = pca.fit_transform(samples)
                    samples_val_embed = pca.transform(samples_val)
                    ref_embed = pca.transform(ref)
                    xtest_embed = pca.transform(X_test)
                elif args.embedding_method == "umap":
                    print(f"Seed {seed}: Using UMAP for embedding...")
                    reducer = umap.UMAP(n_neighbors=15, n_components=64, random_state=seed)
                    samples_embed = reducer.fit_transform(samples)
                    samples_val_embed = reducer.transform(samples_val)
                    ref_embed = reducer.transform(ref)
                    xtest_embed = reducer.transform(X_test)
                else:
                    raise ValueError("Unknown embedding_method.")
            elif args.density_estimator in ["freq", "discrete_kde"]:
                print(f"Seed {seed}: Using raw data (no embedding) for density estimation...")
                samples_embed = samples
                samples_val_embed = samples_val
                ref_embed = ref
                xtest_embed = X_test
            else:
                raise ValueError("Unknown density_estimator method.")

            # -------------------------------
            # Density Estimation Block


            if args.density_estimator == "bnaf":
                print(f"Seed {seed}: Using BNAF-based density estimation...")
                _gen, model_gen = density_estimator_trainer(
                    samples_embed,
                    samples_val_embed[: len(samples_val_embed) // 2],
                    samples_val_embed[len(samples_val_embed) // 2:]
                )
                _data, model_data = density_estimator_trainer(
                    ref_embed,
                    ref_embed[: len(ref_embed) // 2],
                    ref_embed[len(ref_embed) // 2:]
                )

                # Move models to the correct device
                model_gen = model_gen.to(device)
                model_data = model_data.to(device)

                p_G_evaluated = np.exp(
                    compute_log_p_x(model_gen, torch.as_tensor(xtest_embed).float().to(device))
                    .cpu().detach().numpy()
                )
                p_R_evaluated = np.exp(
                    compute_log_p_x(model_data, torch.as_tensor(xtest_embed).float().to(device))
                    .cpu().detach().numpy()
                ) + 1e-30

            elif args.density_estimator == "kde":
                print(f"Seed {seed}: Using Gaussian KDE in PCA/UMAP space...")
                density_gen = stats.gaussian_kde(samples_embed.transpose(1, 0))
                density_data = stats.gaussian_kde(ref_embed.transpose(1, 0))
                p_G_evaluated = density_gen(xtest_embed.transpose(1, 0))
                p_R_evaluated = density_data(xtest_embed.transpose(1, 0)) + 1e-30

            elif args.density_estimator == "freq":
                print(f"Seed {seed}: Computing frequency-based likelihood...")
                pG_func = compute_freq_likelihood(samples_embed)
                p_G_evaluated = np.array([pG_func(x) for x in xtest_embed])
                pR_func = compute_freq_likelihood(ref_embed)
                p_R_evaluated = np.array([pR_func(x) for x in xtest_embed]) + 1e-30

            elif args.density_estimator == "discrete_kde":
                print(f"Seed {seed}: Computing discrete KDE (Hamming-based)...")
                pG_func = discrete_kde(samples_embed, alpha=0.5)
                p_G_evaluated = np.array([pG_func(x) for x in xtest_embed])
                pR_func = discrete_kde(ref_embed, alpha=0.5)
                p_R_evaluated = np.array([pR_func(x) for x in xtest_embed]) + 1e-30

            # -------------------------------
            # Membership Inference Evaluation
            # Equation (1): using p_G(x_i)
            log_p_test = p_G_evaluated
            thres = np.quantile(log_p_test, 0.5)
            auc_y = np.hstack((np.ones(train.shape[0]), np.zeros(test.shape[0])))
            fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_test, pos_label=1)
            eqn1_auc = metrics.auc(fpr, tpr)
            print(f"Seed {seed}: Eqn.(1) AUC: {eqn1_auc}")

            # Equation (2): ratio p_G(x_i)/p_R(x_i)
            # For BNAF and KDE, we already computed p_R_evaluated;
            # if needed, recompute for bnaf below.
            if args.density_estimator == "bnaf":
                p_R_evaluated = np.exp(
                    compute_log_p_x(model_data, torch.as_tensor(xtest_embed).float().to(device))
                    .cpu().detach().numpy()
                )
            p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)
            eqn2_acc, eqn2_auc = compute_metrics_baseline(p_rel, Y_test)
            print(f"Seed {seed}: Eqn.(2) Accuracy: {eqn2_acc}, AUC: {eqn2_auc}")


            baseline_results, baseline_scores = baselines(
                xtest_embed,
                Y_test,
                samples_embed,
                ref_embed if args.density_estimator in ["bnaf", "kde", "umap"] else ref,
                ref_embed if args.density_estimator in ["bnaf", "kde", "umap"] else ref,
            )

            privacy_logger[key_base][f"{seed}_Baselines"] = baseline_results
            privacy_logger[key_base][f"{seed}_BaselineScore"] = baseline_scores
            for attack_name, attack_arr in baseline_scores.items():
                privacy_logger[key_base][f"{seed}_Baseline_{attack_name}Scores"] = attack_arr
            privacy_logger[key_base][f"{seed}_Xtest"] = xtest_embed
            privacy_logger[key_base][f"{seed}_Ytest"] = Y_test
            privacy_logger[key_base][f"{seed}_Eqn1"] = (p_G_evaluated > thres).sum() / len(xtest_embed)
            privacy_logger[key_base][f"{seed}_Eqn1_thres"] = thres
            privacy_logger[key_base][f"{seed}_Eqn1AUC"] = eqn1_auc
            privacy_logger[key_base][f"{seed}_Eqn1Score"] = log_p_test
            privacy_logger[key_base][f"{seed}_Eqn2"] = eqn2_acc
            privacy_logger[key_base][f"{seed}_Eqn2AUC"] = eqn2_auc
            privacy_logger[key_base][f"{seed}_Eqn2Score"] = p_rel

            print("Baselines for seed", seed, ":", privacy_logger[key_base][f"{seed}_Baselines"])
            print("Eqn1 Accuracy:", privacy_logger[key_base][f"{seed}_Eqn1"],
                  "Eqn1 AUC:", privacy_logger[key_base][f"{seed}_Eqn1AUC"])
            print("Eqn2 Accuracy:", privacy_logger[key_base][f"{seed}_Eqn2"],
                  "Eqn2 AUC:", privacy_logger[key_base][f"{seed}_Eqn2AUC"])

 
with open(f'../../results_folder/{args.dataset}/{args.model}/{args.density_estimator}_{args.embedding_method}_privacy_logger.pkl', 'wb') as pf:
    pickle.dump(privacy_logger, pf)