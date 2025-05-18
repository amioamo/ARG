
"""
Based on the implementation of https://github.com/casey-meehan/data-copying/blob/master/data_copying_tests.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
import pickle
import os
import argparse
from sklearn.decomposition import PCA


from typing import Dict



def Zu(Pn, Qm, T):
    """Extracts distances to training nearest neighbor
    L(P_n), L(Q_m), and runs Z-scored Mann Whitney U-test.
    For the global test, this is used on the samples within each cell.

    Inputs:
        Pn: (n * d) np array representing test sample of
            length n (with dimension d)

        Qm: (m * d) np array representing generated sample of
            length m (with dimension d)

        T: (l * d) np array representing training sample of
            length l (with dimension d)

    Ouptuts:
        Zu: Z-scored U value. A large value >>0 indicates
            underfitting by Qm. A small value <<0 indicates.
    """
    m = Qm.shape[0]
    n = Pn.shape[0]

    #fit NN model to training sample to get distances to test and generated samples
    T_NN = NN(n_neighbors = 1).fit(T)
    LQm, _ = T_NN.kneighbors(X = Qm, n_neighbors = 1)
    LPn, _ = T_NN.kneighbors(X = Pn, n_neighbors = 1)

    #Get Mann-Whitney U score and manually Z-score it using the conditions of null hypothesis H_0
    u, _ = mannwhitneyu(LQm, LPn, alternative = 'less')
    mean = (n * m / 2) - 0.5 #0.5 is continuity correction
    std = np.sqrt(n*m*(n + m + 1) / 12)
    Z_u = (u - mean) / std
    return Z_u

def Zu_cells(Pn, Pn_cells, Qm, Qm_cells, T, T_cells):
    """Collects the Zu statistic in each of k cells.
    There should be >0 test (Pn) and train (T) samples in each of the cells.

    Inputs:
        Pn: (n X d) np array representing test sample of length
            n (with dimension d)

        Pn_cells: (1 X n) np array of integers indicating which
            of the k cells each sample belongs to

        Qm: (m X d) np array representing generated sample of
            length n (with dimension d)

        Qm_cells: (1 X m) np array of integers indicating which of the
            k cells each sample belongs to

        T: (l X d) np array representing training sample of
            length l (with dimension d)

        T_cells: (1 X l) np array of integers indicating which of the
            k cells each sample belongs to

    Outputs:
        Zus: length k np array, where entry i indicates the Zu score for cell i
    """
    #assume cells are labeled 0 to k-1
    k = len(np.unique(Pn_cells))
    Zu_cells = np.zeros(k)

    #get samples in each cell and collect Zu
    for i in range(k):
        Pn_cell_i = Pn[Pn_cells == i]
        Qm_cell_i = Qm[Qm_cells == i]
        T_cell_i = T[T_cells == i]
        #check that the cell has test and training samples present
        if len(Pn_cell_i) * len(T_cell_i) == 0:
            raise ValueError("Cell {:n} lacks test samples and/or training samples. Consider reducing the number of cells in partition.".format(i))

        #if there are no generated samples present, add a 0 for Zu. This cell will be excluded in \Pi_\tau
        if len(Qm_cell_i) > 0:
            Zu_cells[i] = Zu(Pn_cell_i, Qm_cell_i, T_cell_i)
        else:
            Zu_cells[i] = 0
            print("cell {:n} unrepresented by Qm".format(i))

    return Zu_cells





def C_T(Pn, Pn_cells, Qm, Qm_cells, T, T_cells, tau):
    """Runs C_T test given samples and their respective cell labels.
    The C_T statistic is a weighted average of the in-cell Zu statistics, weighted
    by the share of test samples (Pn) in each cell. Cells with an insufficient number
    of generated samples (Qm) are not included in the statistic.

    Inputs:
        Pn: (n X d) np array representing test sample of length
            n (with dimension d)

        Pn_cells: (1 X n) np array of integers indicating which
            of the k cells each sample belongs to

        Qm: (m X d) np array representing generated sample of
            length n (with dimension d)

        Qm_cells: (1 X m) np array of integers indicating which of the
            k cells each sample belongs to

        T: (l X d) np array representing training sample of
            length l (with dimension d)

        T_cells: (1 X l) np array of integers indicating which of the
            k cells each sample belongs to

        tau: (scalar between 0 and 1) fraction of Qm samples that a
            cell needs to be included in C_T statistic.

    Outputs:
        C_T: The C_T statistic for the three samples Pn, Qm, T
    """

    m = Qm.shape[0]
    n = Pn.shape[0]
    k = np.max(np.unique(T_cells)) + 1 #number of cells

    #First, determine which of the cells have sufficient generated samples (Qm(pi) > tau)
    labels, cts = np.unique(Qm_cells, return_counts = True)
    Qm_cts = np.zeros(k)
    Qm_cts[labels.astype(int)] = cts #put in order of cell label
    Qm_of_pi = Qm_cts / m
    Pi_tau = Qm_of_pi > tau #binary array selecting which cells have sufficient samples

    #Get the fraction of test samples in each cell Pn(pi)
    labels, cts = np.unique(Pn_cells, return_counts = True)
    Pn_cts = np.zeros(k)
    Pn_cts[labels.astype(int)] = cts #put in order of cell label
    Pn_of_pi = Pn_cts / n

    #Now get the in-cell Zu scores
    Zu_scores = Zu_cells(Pn, Pn_cells, Qm, Qm_cells, T, T_cells)

    #compute C_T:
    C_T = Pn_of_pi[Pi_tau].dot(Zu_scores[Pi_tau])/np.sum(Pn_of_pi[Pi_tau])

    return C_T

def dataset_level_copy(syn, train, test, k):

    """
    An implementation for data-copy based on 'A Non-Parametric Test to Detect Data-Copying in Generative Models'
    """

    # firstly perform PCA

    pca = PCA(n_components=100)

    # Perform PCA on synthetic_set
    syn_pca = pca.fit_transform(syn)
    test_pca = pca.fit_transform(test)
    train_pca = pca.fit_transform(train)


    KM_clf = KMeans(n_clusters= k).fit(train_pca)  # instance space partition
    T_labels = KM_clf.predict(train_pca)  # cell labels for Train
    Pn_label = KM_clf.predict(test_pca)  # cell labels for Test
    Qm_labels = KM_clf.predict(syn_pca) # cell labels for Syn

    return C_T(test_pca, Pn_label, syn_pca, Qm_labels, train_pca, T_labels, tau = 20 / len(syn))




def KING(array1, array2):
    """
    Given two arrays of SNP sequences, return the KING
    """
    qG = np.array(array1)
    dG = np.array(array2)

    n11 = np.sum((qG == 1) & (dG == 1))
    n02 = np.sum((qG == 0) & (dG == 2))
    n20 = np.sum((qG == 2) & (dG == 0))
    n1_ = np.sum(qG == 1)
    n_1 = np.sum(dG == 1)

    if n1_ != 0:
        phi = (2 * n11 - 4 * (n02 + n20) - n_1 + n1_) / (4 * n1_)
    else:
        phi = (2 * n11 - 4 * (n02 + n20) - n_1 + n1_) / (4 * n_1)

    return phi

def record_level_copying(syn, train):
   # if syn.shape != train.shape:
       # print('mismatch dimensions')

        #Typically, size(syn) > size(train)
        # so we randomly select a subset
       # return None, None

    top_indices = []
    top_king_values = []

    for i, syn_record in enumerate(syn):
        king_scores = []
        for j, train_record in enumerate(train):
            king_score = KING(syn_record, train_record)
            king_scores.append((king_score, j))

        # Sort by KING score and take the top 3
        top_3 = sorted(king_scores, reverse=True)[:3]

        # Extract indices and KING values
        indices, values = zip(*top_3)
        top_indices.append(indices)
        top_king_values.append(values)

    return np.array(top_indices), np.array(top_king_values)



def overfit_run(args):
    """
    parameters
    """
    os.makedirs(f"../../results_folder/{args.dataset}/{args.model}", exist_ok=True)
    print('Now evaluate utility for dataset ' + args.dataset)

    datacopying_logger: Dict = {}
    datacopying_logger_ref: Dict = {}
    real_data_file = "../data/" + args.dataset + "/real/"
    syn_data_file = "../data/" + args.dataset + "/syn/" + args.model + "/"




    for snp_num in args.snp_nums:

        for train_raito in args.train_ratios:
            datacopying_logger[
                f"{snp_num}_{train_raito}"
            ] = {}

            datacopying_logger_ref[
                f"{snp_num}_{train_raito}"
            ] = {}

            for i_iter, seed in enumerate(args.seeds):
                print(i_iter, snp_num, train_raito)

                train_data = np.load(real_data_file + str(snp_num) + '/' + \
                               str(train_raito) + '/' + 'train_' + str(seed) + '.npy')


                test_data = np.load(real_data_file + str(snp_num) + '/' + \
                                     str(train_raito) + '/' + 'test_' + str(seed) + '.npy')

                ref_data = np.load(real_data_file + str(snp_num) + '/' + 'ref.npy')

                X_data = np.concatenate([train_data, test_data])


                if args.model == 'ctgan':
                    syn_data = np.load(syn_data_file + str(snp_num) + '/' + \
                               str(train_raito) + '/' + 'sample_data_epoch_2000_' + str(seed) + '.pkl', \
                                   allow_pickle = True).values
                    syn_data[np.where(syn_data < 0)[0], np.where(syn_data < 0)[1]] = 0  # remove the negative values
                    # though do not know why they exist
                else:
                    syn_data = np.load(syn_data_file + str(snp_num) + '/' + \
                                       str(train_raito) + '/' + 'sample_data_epoch_1000_' + str(seed) + '.npy', \
                                       allow_pickle=True)
                ### only using 1000 records
                syn_indices = np.random.choice(syn_data.shape[0], size=1000, replace=False)
                syn_data = syn_data[syn_indices, :]

               #print(dataset_level_copy(ref_data, train_data, test_data, 5))
                datacopying_logger[
                    f"{snp_num}_{train_raito}"
                ][f"{seed}_rlevel"] = record_level_copying(syn_data, train_data)
                datacopying_logger[
                    f"{snp_num}_{train_raito}"
                ][f"{seed}_dlevel"] = dataset_level_copy(syn_data, train_data, ref_data, 3)
                print(datacopying_logger[
                    f"{snp_num}_{train_raito}"
                ][f"{seed}_dlevel"])

                # datacopying_logger_ref[
                #     f"{snp_num}_{train_raito}"
                # ][f"{seed}_rlevel"] = record_level_copying(X_data, syn_data)
                # datacopying_logger_ref[
                #     f"{snp_num}_{train_raito}"
                # ][f"{seed}_dlevel"] = dataset_level_copy(test_data, syn_data, ref_data, 3)
                # print(datacopying_logger_ref[
                #           f"{snp_num}_{train_raito}"
                #       ][f"{seed}_dlevel"])



    # with open(f'../../results_folder/{args.dataset}/{args.model}/overfitting_logger.pkl', 'wb') as pf:
    #     pickle.dump(datacopying_logger, pf)

    with open(f'../results_folder/{args.dataset}/{args.model}/ref_data_overfitting_logger_bio.pkl', 'wb') as pf:
        pickle.dump(datacopying_logger_ref, pf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--snp_nums",
        type=int,
        nargs="+",
        default=[200, 500, 1000],
        help="sequence length"
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
        default='1kgp'
    )

    parser.add_argument(
        "--syn_size",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--model",
        type=str,
        default='ctgan'
    )

    args = parser.parse_args()


    overfit_run(args)

