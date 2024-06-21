import pandas as pd
import numpy as np
import pickle
import zipfile

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import GroupKFold,KFold
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from sklearn import svm
import time

import psutil
import os

from sklearn.metrics import average_precision_score, roc_curve, confusion_matrix, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
device_cpu = device
print(device)

mytype = torch.float16  # to save memory (only on GPU)
mytype = torch.float32

def load_df(name,dataset_dir="data"):
    """
    Loads a dataframe from a CSV file, cleans up SMILES strings that cannot be read by RDKit,
    and returns the cleaned dataframe.

    :param name: The name of the file (with extension) to be loaded. If the file is a zip archive,
                 it will be extracted first.
    :type name: str
    :param dataset_dir: The directory where the dataset is located, by default "data".
    :type dataset_dir: str, optional
    :return: The cleaned dataframe with SMILES strings that RDKit can read.
    :rtype: pd.DataFrame
    """
    # if the data is in a zip file, unzip it
    if ".zip" in name:
        with zipfile.ZipFile(f"{dataset_dir}/{name}", 'r') as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/")
        name = name[:-4]
    df = pd.read_csv(f"{dataset_dir}/{name}",index_col=0)
    # clean smiles
    smiles = df[['SMILES']].drop_duplicates().values.flatten()
    l_smiles = [sm for sm in smiles if Chem.MolFromSmiles(sm) is None]
    print(f"number of smiles to clean: {len(l_smiles)}")
    df = df[~df['SMILES'].isin(l_smiles)]
    print(f"{name} shape",df.shape)
    return df

def add_indsmiles(df):
    """
    Adds a column to the dataframe with indices for each unique SMILES string.

    :param df: The dataframe to be processed.
    :type df: pd.DataFrame
    :return: A tuple containing the processed dataframe with an added column for SMILES indices
             and the array of unique SMILES strings.
    :rtype: tuple(pd.DataFrame, np.ndarray)
    """
    # Index of the smiles in the dataset
    smiles = df[['SMILES']].drop_duplicates().values.flatten()
    nM = len(smiles)
    print("number of different smiles (mol):", nM)
    dict_smiles2ind = {smiles[i]: i for i in range(nM)}
    # Add indsmiles in df
    df['indsmiles'] = df['SMILES'].apply(lambda x: dict_smiles2ind[x])
    df = df.sort_values(by=['indsmiles'])
    df = df.reset_index(drop=True)
    return df, smiles

def Morgan_FP(list_smiles):
    """
    Computes the Morgan fingerprints for a list of SMILES strings.

    :param list_smiles: A list of SMILES strings for which to compute the fingerprints.
    :type list_smiles: list
    :return: A tensor containing the Morgan fingerprints for the input SMILES strings.
    :rtype: torch.Tensor
    """
    ms = [Chem.MolFromSmiles(sm) for sm in list_smiles]
    nM = len(ms)
    MorganFP = np.zeros((nM, 1024))
    for i in range(nM):
        # Generate Morgan fingerprint of the molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, nBits=1024)
        # Convert the fingerprint to a numpy array
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        MorganFP[i, :] = arr
    MorganFP = MorganFP.astype(int)
    MorganFP = torch.tensor(MorganFP, dtype=mytype).to(device)
    return MorganFP

def Nystrom_X_cn(mM,rM,nM,MorganFP):
    """
    Compute the Nystrom approximation of the centered normalized feature matrix.

    :param mM: Number of molecule landmarks
    :type mM: int
    :param rM: Number of dimensions to keep after SVD for molecule features
    :type rM: int
    :param nM: Total number of molecules.
    :type nM: int
    :param MorganFP: Matrix of Morgan fingerprints of shape (nM, fingerprint_length).
    :type MorganFP: numpy.ndarray
    :return: The centered normalized feature matrix.
    :rtype: torch.Tensor
    :notes: This function computes the Nystrom approximation of the feature matrix using the given
            Morgan fingerprints. It first selects a random subset S of size mM from the total
            nM molecules. It then computes the kernel matrix K using the selected subset.
            The approximate feature matrix is computed using Singular Value Decomposition (SVD)
            on K. Finally, the features are normalized by centering and dividing by their L2 norm.

            The input MorganFP should be a numpy array with shape (nM, fingerprint_length).
    """
    S = np.random.permutation(nM)[:mM]
    S = np.sort(S)
    K = ( MorganFP[S,:] @ MorganFP.T ) / ( 1024 - (1-MorganFP[S,:]) @ (1-MorganFP.T) )
    print("mol kernel shape",K.shape)
    # compute the approximate mol features with SVD
    U, Lambda, VT = torch.svd(K[:,S])
    epsi = 1e-8  # be careful when we divide by Lambda near 0
    X = K.T @ U[:,:rM] @ torch.diag(1./torch.sqrt(epsi + Lambda[:rM]))
    # nomramlisation of the features
    X_c = X - X.mean(axis = 0)
    X_cn = X_c / torch.norm(X_c,dim = 1)[:,None]
    return X_cn

def Nystrom_X(smiles_list, S, MorganFP, V, rM, Mu, epsi):
    """
    Computes the approximate features of the molecular kernel using the Nystrom approximation.

    :param smiles_list: A list of SMILES strings for which to compute features.
    :param S: Indices of the subset of samples used for the Nystrom approximation.
    :param MorganFP: Precomputed Morgan Fingerprints for the dataset.
    :param V: Eigenvectors of the kernel matrix.
    :param rM: Rank of the approximation.
    :param Mu: Eigenvalues of the kernel matrix.
    :param epsi: Small regularization term added for numerical stability.
    :return: Approximated feature matrix for the input SMILES list.
    :rtype: torch.Tensor
    """
    # Compute Morgan fingerprints for the input list of SMILES
    MorganFP_list = Morgan_FP(smiles_list)

    # Compute the Nystrom approximation of the molecular kernel and the features
    Z_list = (MorganFP[S, :] @ MorganFP_list.T) / (1024 - (1 - MorganFP[S, :]) @ (1 - MorganFP_list.T))
    print("Z_list shape", Z_list.shape)

    X_list = Z_list.T @ V[:, :rM] @ torch.diag(1. / torch.sqrt(epsi + Mu[:rM]))
    print("mol features list shape", X_list.shape)
    return X_list

def add_indfasta(df):
    """
    Adds an index column for each unique FASTA sequence in the dataframe.

    :param df: Dataframe containing protein sequences.
    :return: The updated dataframe with an 'indfasta' column and the array of unique FASTA sequences.
    :rtype: tuple[pd.DataFrame, np.ndarray]
    """
    # Index of the protein in the dataset
    fasta = df[['Target Sequence']].drop_duplicates().values.flatten()
    print("number of different Fasta (protein):", len(fasta))
    # Add ind_fasta in the dataframe
    df['indfasta'] = df['Target Sequence'].apply(lambda x: np.where(fasta == x)[0][0])
    return df, fasta

def load_datas(df):
    """
    Loads interaction data including indices of molecule and protein pairs along with their labels.

    :param df: Dataframe containing the interaction data.
    :return: Tensors of molecule indices, protein indices, and interaction labels.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    array_df = df[['indfasta', 'indsmiles', 'Label']].to_numpy()
    # Extract indices for proteins, molecules, and labels
    J = array_df[:, 0]  # proteins
    I = array_df[:, 1]  # molecules
    y = array_df[:, 2]  # labels
    I = torch.tensor(I, dtype=torch.long).to(device)
    J = torch.tensor(J, dtype=torch.long).to(device)
    y = torch.tensor(np.sign(y - .5)).to(device)
    return I, J, y

def SVM_bfgs(X_cn, Y_cn, y, I, J, lamb,niter = 50):
    """
    Trains an SVM model using the L-BFGS optimization algorithm to minimize the loss function.

    :param X_cn: Feature matrix for compounds.
    :param Y_cn: Feature matrix for targets.
    :param y: Interaction labels.
    :param I: Indices of molecules.
    :param J: Indices of proteins.
    :param lamb: Regularization parameter.
    :param niter: Number of iterations for the L-BFGS optimization algorithm.
    :return: Optimized weight and bias parameters for the SVM model, and the history of the loss function.
    :rtype: tuple[torch.Tensor, torch.Tensor, list[float]]
    """
    n = len(I)
    XI = X_cn[I, :]
    def U(w): return torch.sum((Y_cn @ w)[J, :] * XI, axis=1)  # FAST

    def Loss(u): return 1 / n * torch.sum(torch.maximum(1 + u, torch.tensor(0)))  # Loss function
    def g(w, b): return Loss(-y * (U(w) + b)) + lamb / 2 * (w ** 2).sum()  # Function to minimize

    # L-BFGS optimization
    def closure():
        lbfgs.zero_grad()
        objective = g(w_bfgs, b_bfgs)
        objective.backward()
        return objective

    rM = X_cn.shape[1]
    rP = Y_cn.shape[1]

    w_bfgs = torch.randn(rP, rM).to(device)
    b_bfgs = torch.randn(1).to(device)
    w_bfgs.requires_grad = True
    b_bfgs.requires_grad = True

    lbfgs = optim.LBFGS([w_bfgs, b_bfgs], history_size=10, max_iter=4, line_search_fn="strong_wolfe")
    
    history_lbfgs = []
    tic = time.perf_counter()
    for i in range(niter):
        history_lbfgs.append(g(w_bfgs, b_bfgs).item())
        lbfgs.step(closure)
    print(f"L-BFGS time: {time.perf_counter() - tic:0.4f} seconds")
    return w_bfgs, b_bfgs,history_lbfgs

def compute_proba_Platt_Scalling(w_bfgs, X_cn, Y_cn, y, I, J,niter = 20):
    """
    Computes probability estimates for interaction predictions using Platt scaling.

    :param w_bfgs: Optimized weights from the SVM model.
    :param X_cn: Feature matrix for compounds.
    :param Y_cn: Feature matrix for targets.
    :param y: Interaction labels.
    :param I: Indices of molecules.
    :param J: Indices of proteins.
    :param niter: Number of iterations for the L-BFGS optimization algorithm.
    :return: Optimized parameters 's' and 't' for Platt scaling, and the history of the loss function.
    :rtype: tuple[torch.Tensor, torch.Tensor,list[float]]
    """
    n = len(I)
    XI = X_cn[I, :]
    def U(w): return torch.sum((Y_cn @ w)[J, :] * XI, axis=1)  # FAST
    m = U(w_bfgs).detach()  # Detach to stop backpropagation

    # Logistic regression to find 's' and 't'
    def E(s, t): return 1 / n * torch.sum(torch.log(1 + torch.exp(-y * (s * m + t))))

    # L-BFGS optimization
    def closure():
        lbfgs.zero_grad()
        objective = E(s, t)
        objective.backward()
        return objective
    s = torch.ones(1).to(device)
    t = torch.zeros(1).to(device)
    s.requires_grad = True
    t.requires_grad = True
    lbfgs = optim.LBFGS([s, t], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    history_lbfgs = []
    for i in range(niter):
        history_lbfgs.append(E(s, t).item())
        lbfgs.step(closure)

    return s, t,history_lbfgs


def compute_proba(w_bfgs, b_bfgs, s, t, X_cn, Y_cn, I, J):
    """
    Computes probabilities using the trained weights and Platt scaling parameters.

    :param w_bfgs: Optimized weights from the SVM model for compounds.
    :param b_bfgs: Optimized bias from the SVM model.
    :param s: Optimized scaling parameter from Platt scaling.
    :param t: Optimized offset parameter from Platt scaling.
    :param X_cn: Feature matrix for compounds.
    :param Y_cn: Feature matrix for targets.
    :param I: Indices of molecules.
    :param J: Indices of proteins.
    :return: A tuple containing the margin values, predicted labels, and predicted probabilities.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    m = torch.sum((Y_cn @ w_bfgs)[J, :] * X_cn[I, :], axis=1)  # Margin computation
    y_pred = torch.sign(m + b_bfgs).detach()  # Label prediction
    proba_pred = torch.sigmoid(s * m + t).detach()  # Probability prediction
    return m, y_pred, proba_pred

def results(y, y_pred, proba_pred):
    """
    Computes and returns various performance metrics of the model including accuracy, AUC, average precision,
    optimal threshold for classification, accuracy at the best threshold, confusion matrix, and false positive rate.

    :param y: True labels.
    :param y_pred: Predicted labels before applying Platt scaling.
    :param proba_pred: Predicted probabilities after applying Platt scaling.
    :return: A tuple containing accuracy before Platt scaling, AUC, average precision, optimal threshold,
             accuracy at optimal threshold, confusion matrix, and false positive rate at the optimal threshold.
    :rtype: tuple[float, float, float, float, float, np.ndarray, float]
    """
    # Accuracy before Platt scaling
    acc1 = torch.sum(y == y_pred) / len(y)

    # AUC computation
    fpr, tpr, thresholds = roc_curve(y.cpu().numpy(), proba_pred.cpu().numpy())
    au_Roc = auc(fpr, tpr)

    # Average precision computation
    au_PR = average_precision_score(y.cpu().numpy(), proba_pred.cpu().numpy())

    # Optimal threshold for classification
    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    # Accuracy at the optimal threshold
    y_pred_s = [1 if i else -1 for i in (proba_pred >= thred_optim)]
    acc_best = torch.sum(y == torch.tensor(y_pred_s).to(y.device)) / len(y)

    # Confusion matrix computation
    cm = confusion_matrix(y.cpu().numpy(), y_pred_s)

    # False positive rate at the optimal threshold
    FP = cm[0, 1] / (cm[0, 1] + cm[0, 0])

    return acc1.item(), au_Roc, au_PR, thred_optim, acc_best.item(), cm, FP

 
def make_train_test_val_S1(df,train_ratio=0.8,test_ratio=0.12):
    """
    Splits the input DataFrame into training, testing, and validation datasets. The function first converts the DataFrame into a 
    numpy matrix where rows correspond to 'indfasta' (protein indices), columns to 'indsmiles' (drug indices), and cell values to 'score' (interaction score).
    It then identifies positive (interaction score = 1) and negative (interaction score = 0) interactions and distributes them into 
    training, testing, and validation sets according to the specified ratios. The split is done ensuring that each set contains a 
    balanced proportion of positive and unknown interactions.

    :param df: Input data containing the columns 'indfasta', 'indsmiles', and 'score'.
    :type df: pandas.DataFrame
    :param train_ratio: The proportion of the dataset to be used for the training set, defaults to 0.8.
    :type train_ratio: float, optional
    :param test_ratio: The proportion of the dataset to be used for the testing set, defaults to 0.12.
    :type test_ratio: float, optional
    :return: A tuple of numpy arrays representing the training, testing, and validation datasets respectively.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    Note:
    The remaining portion of the dataset not allocated to training or testing is used for validation.
    This function assumes that the DataFrame's 'score' column contains binary values (1 for interaction and 0 for no interaction).
    NaN values in 'score' are treated as unknown interactions and are handled separately.
    """

    try : 
        intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    except:
        intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    n_p,n_m = intMat.shape # number of proteins and drugs
    Ip, Jm = np.where(intMat==1) # indices of interactions +
    nb_positive_inter = int(len(Ip))
    Inp, Jnm = np.where(intMat==0)
    Inkp, Jnkm = np.where(np.isnan(intMat))

    S = np.random.permutation(nb_positive_inter) # shuffle the indices of interactions +
    train_index = S[:int(train_ratio*nb_positive_inter)]
    test_index = S[int(train_ratio*nb_positive_inter):int((train_ratio+test_ratio)*nb_positive_inter)]
    val_index = S[int((train_ratio+test_ratio)*nb_positive_inter):]
    print("train", len(train_index), "test", len(test_index), "val", len(val_index))

    #### TRAIN ####
    Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

    Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

    train = np.zeros([1,3], dtype=int)

    nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
    for i in range(nb_prot):

        j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

        indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
        indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
        indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

        indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
        indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

        nb_positive_interactions = len(indice_P)
        nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
        indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
        indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

        if len(indice_P) <= len(indice_freq_one_prot):
            # we shoot at random nb_positive_interactions in drugs with a lot of interactions
            indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                len(indice_P), replace = False)
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
            # we shoot at random nb_positive_interactions in drugs with a lot of interactions
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_N_one_prot_poss))
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
            # we shoot at random nb_positive_interactions in drugs with a lot of interactions
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_N_one_prot_poss))
        else:
            # we shoot at random nb_positive_interactions in drugs with a lot of interactions
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
            #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
            #print(indice_poss_one_prot_NK.shape)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

        Mp[indice_N_one_prot.astype(int)]-=1

        # this protein has been processed
        Mm[j] = 0

        indice = np.r_[indice_P,indice_N_one_prot].astype(int)
        etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
        A = np.stack((indice, etiquette), axis=-1)
        B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
        train = np.concatenate((train,B))

    train = train[1:]
    print("train", train.shape)

    ##### TEST ####
    # interactions + in test
    indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
    print("nb of interactions + in test",len(indice_P_t))

    # interactions - in test
    a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
    a1 = set(map(tuple, a))
    b = train[:,:2]   # all the interactions in the train
    b1 = set(map(tuple, b))
    indice_N_t = np.array(list(a1 - b1))#[:indice_P_t.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
    print("number of real interactions - in test",len(indice_N_t))

    # add interactions np.nan in test

    if len(indice_N_t) == 0:
        # initialization
        indice_N_t = np.array([-1, -1]).reshape(1,2)

    c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

    if len(indice_N_t) < indice_P_t.shape[0]:
        # we add some interactions - in test to have the same number of interactions + and - in test choose in the np.nan in the matrix
        k = 0
        while len(indice_N_t) < indice_P_t.shape[0]+1:
            i = np.random.randint(0, len(c))
            if tuple(c[i]) not in b1:
                indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
                k += 1

    # we drop the first row of indice_N_t if is [-1, -1]
    if indice_N_t[0,0] == -1:
        indice_N_t = indice_N_t[1:,:]

    indice_N_t = indice_N_t[:len(indice_P_t),:]
    print("number of interactions - in test",len(indice_N_t))
    # we add the column of 0 for the etiquette
    indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
    test = np.r_[indice_P_t,indice_N_t]
    print("test", test.shape)

    ##### VALIDATION ####
    # interactions + in val
    indice_P_v = np.c_[Ip[val_index],Jm[val_index], np.ones(len(val_index))].astype(int)
    print("nb of interactions + in val",len(indice_P_v))

    # interactions - in val
    a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
    a1 = set(map(tuple, a))
    b = train[:,:2]   # all the interactions in the train
    b1 = set(map(tuple, b))
    c = test[:,:2]   # all the interactions in the test
    c1 = set(map(tuple, c))
    indice_N_v = np.array(list(a1 - b1 - c1))#[:indice_P_v.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
    print("number of real interactions - in val",len(indice_N_v))

    # add interactions np.nan in val

    if len(indice_N_v) == 0:
        # initialization
        indice_N_v = np.array([-1, -1]).reshape(1,2)

    d = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

    if len(indice_N_v) < indice_P_v.shape[0]:
        # we add some interactions - in val to have the same number of interactions + and - in val choose in the np.nan in the matrix
        k = 0
        while len(indice_N_v) < indice_P_v.shape[0]+1:
            i = np.random.randint(0, len(d))
            if (tuple(d[i]) not in b1) and (tuple(d[i]) not in c1):
                indice_N_v = np.concatenate((indice_N_v, d[i].reshape(1,2)))
                k += 1
    
    # we drop the first row of indice_N_v if is [-1, -1]
    if indice_N_v[0,0] == -1:
        indice_N_v = indice_N_v[1:,:]

    indice_N_v = indice_N_v[:len(indice_P_v),:]
    print("number of interactions - in val",len(indice_N_v))
    # we add the column of 0 for the etiquette
    indice_N_v = np.c_[indice_N_v, np.zeros(len(indice_N_v))].astype(int)
    val = np.r_[indice_P_v,indice_N_v]
    print("val", val.shape)

    print("Train/test/val datasets prepared.")

    return train,test,val

def make_train_test_val_S2(df):
    """
    Splits the interaction data into distinct training, testing, and validation datasets,
    ensuring molecules in the test set are not in the training set and molecules in the validation set
    are not in the training set. It processes input data to form interaction matrices and categorizes
    these interactions into positive, negative, and unknown based on their presence, absence, or uncertainty in the dataset.

    :param df: The input data containing the columns 'indfasta', 'indsmiles', and 'score', representing interaction data between proteins and molecules.
    :type df: pandas.DataFrame
    :return: A tuple of numpy.ndarrays representing the training, testing, and validation datasets. Each set contains arrays of interactions labeled with indices and interaction scores, with drugs and proteins split according to specified criteria to ensure separation between training, testing, and validation sets.
    :rtype: tuple

    The function performs the following steps:
    - Convert the DataFrame to a numpy array of interaction scores.
    - Determine the unique proteins and drugs, assigning them to train, test, and validation groups based on specified proportions.
    - For each group, create datasets of positive and negative interactions, handling missing values as unknown interactions.
    - Ensure there is no overlap of molecules between the training set and either the testing or validation sets.
    - Return the prepared datasets for further processing or model training.
    """

    try : 
        intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    except:
        intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    n_p,n_m = intMat.shape
    Ip, Jm = np.where(intMat==1)  # interactions + in train
    Inp, Jnm = np.where(intMat==0)  # interactions - in train
    Inkp, Jnkm = np.where(np.isnan(intMat)) # interactions np.nan in train


    nP = df[["indfasta"]].drop_duplicates().reset_index().shape[0]
    nM = df[["indsmiles"]].drop_duplicates().reset_index().shape[0]

    SP = np.random.permutation(nP)
    SM = np.random.permutation(nM)

    groups = []
    for ip,im in zip(Ip,Jm):
        if  (im in SM[:int(0.78*nM)]):
            groups.append("train")
        elif (im in SM[int(0.78*nM):int(0.92*nM)]):
            groups.append("test")
        elif (im in SM[int(0.92*nM):]):
            groups.append("val")
        else:
            groups.append("other")

    train_index = np.where(np.array(groups)=="train")[0]
    test_index = np.where(np.array(groups)=="test")[0]
    val_index = np.where(np.array(groups)=="val")[0]

    #### TRAIN ####
    Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning
    Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

    train = np.zeros([1,3], dtype=int)

    nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
    for i in range(nb_prot):

        j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

        indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
        indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
        indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

        indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
        indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

        nb_positive_interactions = len(indice_P)
        nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
        indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
        indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

        if len(indice_P) <= len(indice_freq_one_prot):
            # we shoot at random interactions - for drugs with a lot of interactions +
            indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                len(indice_P), replace = False)
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
            # we shoot at random interactions - for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_N_one_prot_poss))
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
            # we shoot at random interactions np.nan for drugs with a lot of interactions +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_N_one_prot_poss))
        else:
            # we shoot at random interactions np.nan for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
            #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) 
            #print(indice_poss_one_prot_NK.shape)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

        Mp[indice_N_one_prot.astype(int)]-=1

        # this protein has been processed
        Mm[j] = 0

        indice = np.r_[indice_P,indice_N_one_prot].astype(int)
        etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
        A = np.stack((indice, etiquette), axis=-1)
        B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
        train = np.concatenate((train,B))

    train = train[1:]
    print("train", train.shape)

    ##### TEST ####
    # interactions + in test
    indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
    print("nb of interactions + in test",len(indice_P_t))
    I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[1]==x[1]]
    print("number of interactions + deleted in test", len(set(I_t)))
    indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_t))

    # interactions - in test
    a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
    print("number of interactions -", a.shape)
    indice_N_t = np.array([-1, -1]).reshape(1,2)

    S_a = np.random.permutation(len(a))
    for i in S_a[:len(a)*2//3]:
        if  (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break
    
    # add interactions np.nan in test
    c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
    print("number of np.nan", c.shape)
    S_c = np.random.permutation(len(c))

    for i in S_c:
        if (c[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_c = i
            print("i_end", i_end_c)
            break

    indice_N_t = indice_N_t[:len(indice_P_t),:] #
    print("number of interactions - in test",len(indice_N_t))
    # we add the column of 0 for the etiquette
    indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
    test = np.r_[indice_P_t,indice_N_t]
    print("test", test.shape)

    ##### VALIDATION ####
    # interactions + in val
    indice_P_v = np.c_[Ip[val_index],Jm[val_index], np.ones(len(val_index))].astype(int)
    print("nb of interactions + in val",len(indice_P_v))
    I_v = [i for i,elt in enumerate(indice_P_v) for x in train if elt[1]==x[1]]
    print("number of interactions + deleted in test", len(set(I_v)))
    indice_P_v = np.delete(indice_P_v, list(set(I_v)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_v))

    # interactions - in val
    indice_N_v = np.array([-1, -1]).reshape(1,2)

    try:
        i_end_a = i_end_a
    except:
        i_end_a = len(a)*2//3

    for i in S_a[i_end_a+1:]:
        if (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, a[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break

    # add interactions np.nan in val

    if len(indice_N_v) == 0:
        # initialization
        indice_N_v = np.array([-1, -1]).reshape(1,2)
    
    for i in S_c[i_end_c+1:]:
        if  (c[i,1] not in train[:,1]): #we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, c[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            print("i_end_val",i)
            break
    
    # we drop the first row of indice_N_v if is [-1, -1]
    if indice_N_v[0,0] == -1:
        indice_N_v = indice_N_v[1:,:]

    indice_N_v = indice_N_v[:len(indice_P_v),:]
    print("number of interactions - in val",len(indice_N_v))
    # we add the column of 0 for the etiquette
    indice_N_v = np.c_[indice_N_v, np.zeros(len(indice_N_v))].astype(int)
    val = np.r_[indice_P_v,indice_N_v]
    print("val", val.shape)

    print("Train/test/val datasets prepared.")

    return train,test,val

def make_train_test_val_S3(df):
    """
    Splits the interaction matrix into training, testing, and validation sets, ensuring there is no overlap
    between the proteins in the training set and those in the testing or validation sets. The split is based
    on the initial distribution of proteins and molecules in the interaction matrix.

    :param df: The input DataFrame containing the columns 'indfasta' for proteins, 'indsmiles' for molecules, and 'score' for their interaction scores.
    :type df: pandas.DataFrame
    :return: A tuple containing the training, testing, and validation sets, each as a numpy.ndarray with the first column representing the index of proteins, the second column the index of molecules, and the third column the interaction scores (1 for interaction, 0 for no interaction, and np.nan for unknown interactions).
    :rtype: tuple

    The function performs the following operations:
    - Converts the DataFrame to a numpy array representing the interaction scores.
    - Randomly shuffles and splits proteins and molecules into distinct groups for training, testing, and validation based on predefined ratios.
    - Creates interaction datasets for each set, ensuring proteins in the test set are not in the train set, and proteins in the val set are not in the train set, thereby preventing data leakage.
    - Handles missing values and ensures the final datasets are balanced in terms of positive and negative interactions.
    """

    try : 
        intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    except:
        intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    n_p,n_m = intMat.shape
    Ip, Jm = np.where(intMat==1)  # interactions + in train
    Inp, Jnm = np.where(intMat==0)  # interactions - in train
    Inkp, Jnkm = np.where(np.isnan(intMat)) # interactions np.nan in train


    nP = df[["indfasta"]].drop_duplicates().reset_index().shape[0]
    nM = df[["indsmiles"]].drop_duplicates().reset_index().shape[0]

    SP = np.random.permutation(nP)
    SM = np.random.permutation(nM)

    groups = []
    for ip,im in zip(Ip,Jm):
        if (ip in SP[:int(0.74*nP)]):
            groups.append("train")
        elif (ip in SP[int(0.74*nP):int(0.915*nP)]):
            groups.append("test")
        elif (ip in SP[int(0.915*nP):]):
            groups.append("val")
        else:
            groups.append("other")

    train_index = np.where(np.array(groups)=="train")[0]
    test_index = np.where(np.array(groups)=="test")[0]
    val_index = np.where(np.array(groups)=="val")[0]


    
    #### TRAIN ####
    Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning
    Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

    train = np.zeros([1,3], dtype=int)

    nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
    for i in range(nb_prot):

        j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

        indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
        indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
        indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

        indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
        indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

        nb_positive_interactions = len(indice_P)
        nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
        indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
        indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

        if len(indice_P) <= len(indice_freq_one_prot):
            # we shoot at random interactions - for drugs with a lot of interactions +
            indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                len(indice_P), replace = False)
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
            # we shoot at random interactions - for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_N_one_prot_poss))
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
            # we shoot at random interactions np.nan for drugs with a lot of interactions +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_N_one_prot_poss))
        else:
            # we shoot at random interactions np.nan for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
            #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) 
            #print(indice_poss_one_prot_NK.shape)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

        Mp[indice_N_one_prot.astype(int)]-=1

        # this protein has been processed
        Mm[j] = 0

        indice = np.r_[indice_P,indice_N_one_prot].astype(int)
        etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
        A = np.stack((indice, etiquette), axis=-1)
        B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
        train = np.concatenate((train,B))

    train = train[1:]
    print("train", train.shape)

    ##### TEST ####
    # interactions + in test
    indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
    print("nb of interactions + in test",len(indice_P_t))
    I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[0]==x[0]]
    print("number of interactions + deleted in test", len(set(I_t)))
    indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_t))

    # interactions - in test
    a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
    print("number of interactions -", a.shape)
    indice_N_t = np.array([-1, -1]).reshape(1,2)

    S_a = np.random.permutation(len(a))
    for i in S_a[:len(a)*2//3]:
        if (a[i,0] not in train[:,0]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break
    
    # add interactions np.nan in test
    c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
    print("number of np.nan", c.shape)
    S_c = np.random.permutation(len(c))

    for i in S_c:
        if (c[i,0] not in train[:,0]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_c = i
            print("i_end", i_end_c)
            break

    indice_N_t = indice_N_t[:len(indice_P_t),:] #
    print("number of interactions - in test",len(indice_N_t))
    # we add the column of 0 for the etiquette
    indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
    test = np.r_[indice_P_t,indice_N_t]
    print("test", test.shape)

    ##### VALIDATION ####
    # interactions + in val
    indice_P_v = np.c_[Ip[val_index],Jm[val_index], np.ones(len(val_index))].astype(int)
    print("nb of interactions + in val",len(indice_P_v))
    I_v = [i for i,elt in enumerate(indice_P_v) for x in train if elt[0]==x[0]]
    print("number of interactions + deleted in test", len(set(I_v)))
    indice_P_v = np.delete(indice_P_v, list(set(I_v)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_v))

    # interactions - in val
    indice_N_v = np.array([-1, -1]).reshape(1,2)

    try:
        i_end_a = i_end_a
    except:
        i_end_a = len(a)*2//3

    for i in S_a[i_end_a+1:]:
        if (a[i,0] not in train[:,0]): # we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, a[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break

    # add interactions np.nan in val

    if len(indice_N_v) == 0:
        # initialization
        indice_N_v = np.array([-1, -1]).reshape(1,2)
    
    for i in S_c[i_end_c+1:]:
        if (c[i,0] not in train[:,0]): #we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, c[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            print("i_end_val",i)
            break
    
    # we drop the first row of indice_N_v if is [-1, -1]
    if indice_N_v[0,0] == -1:
        indice_N_v = indice_N_v[1:,:]

    indice_N_v = indice_N_v[:len(indice_P_v),:]
    print("number of interactions - in val",len(indice_N_v))
    # we add the column of 0 for the etiquette
    indice_N_v = np.c_[indice_N_v, np.zeros(len(indice_N_v))].astype(int)
    val = np.r_[indice_P_v,indice_N_v]
    print("val", val.shape)

    print("Train/test/val datasets prepared.")

    return train,test,val

def make_train_test_val_S4(df):
    """
    Splits the interaction matrix into balanced training, testing, and validation sets ensuring there is no 
    overlap between the proteins and molecules in the training set compared to those in the testing and 
    validation sets. This aims to create distinct and balanced sets for robust model evaluation.

    :param df: Input DataFrame containing interaction data with 'indfasta' for proteins, 'indsmiles' for molecules, 
               and 'score' for interaction scores.
    :type df: pandas.DataFrame
    :return: A tuple containing the training, testing, and validation datasets. Each dataset consists of a numpy array with three columns: protein indices, molecule indices, and interaction scores. The interaction scores are 1 for positive interactions, 0 for negative interactions, and np.nan for unknown interactions.
    :rtype: tuple

    The function executes the following major steps:
    - Converts the DataFrame to a numpy array to represent the interaction scores.
    - Splits proteins and molecules into distinct groups for training, testing, and validation based on predefined criteria, ensuring there's no overlap between the sets for proteins and molecules.
    - Constructs interaction datasets for each set by maintaining a balance between positive and negative interactions and properly handling unknown interactions.
    - Ensures the testing and validation sets are balanced and do not contain any proteins or molecules present in the training set, thus avoiding data leakage and ensuring the model's generalizability.
    """

    try : 
        intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    except:
        intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    n_p,n_m = intMat.shape
    Ip, Jm = np.where(intMat==1)  # interactions + in train
    Inp, Jnm = np.where(intMat==0)  # interactions - in train
    Inkp, Jnkm = np.where(np.isnan(intMat)) # interactions np.nan in train


    nP = df[["indfasta"]].drop_duplicates().reset_index().shape[0]
    nM = df[["indsmiles"]].drop_duplicates().reset_index().shape[0]

    SP = np.random.permutation(nP)
    SM = np.random.permutation(nM)

    groups = []
    for ip,im in zip(Ip,Jm):
        if (ip in SP[:int(0.76*nP)]) and (im in SM[:int(0.4*nM)]):
            groups.append("train")
        elif (ip in SP[int(0.76*nP):int(0.9*nP)]) and (im in SM[int(0.4*nM):int(0.75*nM)]):
            groups.append("test")
        elif (ip in SP[int(0.9*nP):]) and (im in SM[int(0.75*nM):]):
            groups.append("val")
        else:
            groups.append("other")

    train_index = np.where(np.array(groups)=="train")[0]
    test_index = np.where(np.array(groups)=="test")[0]
    val_index = np.where(np.array(groups)=="val")[0]


    
    #### TRAIN ####
    Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning
    Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

    train = np.zeros([1,3], dtype=int)

    nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
    for i in range(nb_prot):

        j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

        indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
        indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
        indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

        indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
        indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

        nb_positive_interactions = len(indice_P)
        nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

        indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
        indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
        indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
        indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

        if len(indice_P) <= len(indice_freq_one_prot):
            # we shoot at random interactions - for drugs with a lot of interactions +
            indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                len(indice_P), replace = False)
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
            # we shoot at random interactions - for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_N_one_prot_poss))
        elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
            # we shoot at random interactions np.nan for drugs with a lot of interactions +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
            indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_N_one_prot_poss))
        else:
            # we shoot at random interactions np.nan for drugs with 1 interaction +
            nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
            #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) 
            #print(indice_poss_one_prot_NK.shape)
            indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                    nb_negative_interactions_remaining, replace = False )
            indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                            indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

        Mp[indice_N_one_prot.astype(int)]-=1

        # this protein has been processed
        Mm[j] = 0

        indice = np.r_[indice_P,indice_N_one_prot].astype(int)
        etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
        A = np.stack((indice, etiquette), axis=-1)
        B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
        train = np.concatenate((train,B))

    train = train[1:]
    print("train", train.shape)

    ##### TEST ####
    # interactions + in test
    indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
    print("nb of interactions + in test",len(indice_P_t))
    I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[0]==x[0] or elt[1]==x[1]]
    print("number of interactions + deleted in test", len(set(I_t)))
    indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_t))

    # interactions - in test
    a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
    print("number of interactions -", a.shape)
    indice_N_t = np.array([-1, -1]).reshape(1,2)

    S_a = np.random.permutation(len(a))
    for i in S_a[:len(a)*2//3]:
        if (a[i,0] not in train[:,0]) and (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break
    
    # add interactions np.nan in test
    c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
    print("number of np.nan", c.shape)
    S_c = np.random.permutation(len(c))

    for i in S_c:
        if (c[i,0] not in train[:,0]) and (c[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_c = i
            print("i_end", i_end_c)
            break

    indice_N_t = indice_N_t[:len(indice_P_t),:] #
    print("number of interactions - in test",len(indice_N_t))
    # we add the column of 0 for the etiquette
    indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
    test = np.r_[indice_P_t,indice_N_t]
    print("test", test.shape)

    ##### VALIDATION ####
    # interactions + in val
    indice_P_v = np.c_[Ip[val_index],Jm[val_index], np.ones(len(val_index))].astype(int)
    print("nb of interactions + in val",len(indice_P_v))
    I_v = [i for i,elt in enumerate(indice_P_v) for x in train if elt[0]==x[0] or elt[1]==x[1]]
    print("number of interactions + deleted in test", len(set(I_v)))
    indice_P_v = np.delete(indice_P_v, list(set(I_v)) ,axis = 0)
    print("number of interactions + in test", len(indice_P_v))

    # interactions - in val
    indice_N_v = np.array([-1, -1]).reshape(1,2)

    try:
        i_end_a = i_end_a
    except:
        i_end_a = len(a)*2//3

    for i in S_a[i_end_a+1:]:
        if (a[i,0] not in train[:,0]) and (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, a[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break

    # add interactions np.nan in val

    if len(indice_N_v) == 0:
        # initialization
        indice_N_v = np.array([-1, -1]).reshape(1,2)
    
    for i in S_c[i_end_c+1:]:
        if (c[i,0] not in train[:,0]) and (c[i,1] not in train[:,1]): #we drop the interactions- in train and the prot in train
            indice_N_v = np.concatenate((indice_N_v, c[i].reshape(1,2)))
        if len(indice_N_v) == indice_P_v.shape[0] + 1:
            print("i_end_val",i)
            break
    
    # we drop the first row of indice_N_v if is [-1, -1]
    if indice_N_v[0,0] == -1:
        indice_N_v = indice_N_v[1:,:]

    indice_N_v = indice_N_v[:len(indice_P_v),:]
    print("number of interactions - in val",len(indice_N_v))
    # we add the column of 0 for the etiquette
    indice_N_v = np.c_[indice_N_v, np.zeros(len(indice_N_v))].astype(int)
    val = np.r_[indice_P_v,indice_N_v]
    print("val", val.shape)

    print("Train/test/val datasets prepared.")

    return train,test,val

def make_CV_train_test_full(df,nb_folds,path_mkdir):
  """
    Splits the input DataFrame into cross-validation training and testing datasets, ensuring the same proportion 
    of positive (interaction score = 1) and negative (interaction score = 0) interactions in each set. The function 
    performs K-fold cross-validation with the specified number of folds.

    :param df: Input data containing the columns 'indfasta', 'indsmiles', and 'score'.
    :type df: pandas.DataFrame
    :param nb_folds: Number of folds for cross-validation.
    :type nb_folds: int
    :param path_mkdir: Path to the directory where the output train and test sets will be saved as pickle files.
    :type path_mkdir: str
    :return: A tuple containing lists of DataFrames representing the training and testing datasets for each fold.
    :rtype: (list[pandas.DataFrame], list[pandas.DataFrame])

    Note:
    The function first converts the DataFrame into a numpy matrix where rows correspond to 'indfasta' (protein indices), 
    columns to 'indsmiles' (drug indices), and cell values to 'score' (interaction score). Positive (interaction score = 1) 
    and negative (interaction score = 0) interactions are identified and distributed into the folds. NaN values in 'score' 
    are treated as unknown interactions and handled separately. The function then constructs balanced training and testing
    datasets for each fold, ensuring that the same proportion of positive and negative interactions is maintained in both sets.
  """

  try : 
    intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
  except:
    intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)


  # Set the different folds
  skf_positive = KFold(shuffle=True, n_splits=nb_folds)

  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)
  nb_positive_inter = int(len(Ip))
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  for train_index, test_index in skf_positive.split(range(nb_positive_inter)):
      train_index = np.random.choice(train_index, int(len(train_index)), replace=False)

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      # test
      test_index =  np.random.choice(test_index, int(len(test_index)), replace=False)
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
      a1 = set(map(tuple, a))
      b = train[:,:2]   # all the interactions in the train
      b1 = set(map(tuple, b))
      indice_N_t = np.array(list(a1 - b1))#[:indice_P_t.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
      #print(len(indice_N_t))

      # add interactions np.nan in test

      if len(indice_N_t) == 0:
          # initialization
          indice_N_t = np.array([-1, -1]).reshape(1,2)

      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

      if len(indice_N_t) < indice_P_t.shape[0]:
          # we add some interactions - in test to have the same number of interactions + and - in test choose in the np.nan in the matrix
          k = 0
          while len(indice_N_t) < indice_P_t.shape[0]+1:
              i = np.random.randint(0, len(c))
              if tuple(c[i]) not in b1:
                  indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
                  k += 1

      # we drop the first row of indice_N_t if is [-1, -1]
      if indice_N_t[0,0] == -1:
          indice_N_t = indice_N_t[1:,:]

      indice_N_t = indice_N_t[:len(indice_P_t),:]

      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]

      all_test_interactions_arr.append(test)
      print("test", test.shape)

  train_arr = []
  for i in range(len(all_train_interactions_arr)):
    df_train = pd.DataFrame(all_train_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_train_S = df_train.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_train_S = df_train_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_train_S = df_train_S[["smiles","fasta","label"]]
    df_train_S.columns = ["SMILES","Target Sequence","Label"]
    train_arr.append(df_train_S)

  with open(path_mkdir+"/train_arr.pkl","wb") as f:
    pickle.dump(train_arr,f)

  test_arr = []
  for i in range(len(all_test_interactions_arr)):
    df_test = pd.DataFrame(all_test_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_test_S = df_test.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_test_S = df_test_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_test_S = df_test_S[["smiles","fasta","label"]]
    df_test_S.columns = ["SMILES","Target Sequence","Label"]
    test_arr.append(df_test_S)

  with open(path_mkdir+"/test_arr.pkl","wb") as f:
    pickle.dump(test_arr,f)

  print("Train/test datasets prepared.")
  return train_arr, test_arr

def make_CV_train_test_unseen_drug(df,nb_folds,path_mkdir):
  """
    Splits the input DataFrame into cross-validation training and testing datasets, ensuring that the molecules in 
    the test set are not present in the train set. The function performs K-fold cross-validation with the specified 
    number of folds.

    :param df: Input data containing the columns 'indfasta', 'indsmiles', and 'score'.
    :type df: pandas.DataFrame
    :param nb_folds: Number of folds for cross-validation.
    :type nb_folds: int
    :param path_mkdir: Path to the directory where the output train and test sets will be saved as pickle files.
    :type path_mkdir: str
    :return: A tuple containing lists of DataFrames representing the training and testing datasets for each fold.
    :rtype: (list[pandas.DataFrame], list[pandas.DataFrame])

    Note:
    The function first converts the DataFrame into a numpy matrix where rows correspond to 'indfasta' (protein indices), 
    columns to 'indsmiles' (drug indices), and cell values to 'score' (interaction score). Positive (interaction score = 1) 
    and negative (interaction score = 0) interactions are identified and distributed into the folds. NaN values in 'score' 
    are treated as unknown interactions and handled separately. The folds are created such that the molecules in the test 
    set are not present in the train set.
  """
  # Convert the DataFrame to a numpy matrix
  try : 
    intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
  except:
    intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

  # Set up cross-validation
  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  groups = np.array(Jm) #folds are made on molecules
  group_kfold = GroupKFold(n_splits=5)

  nb_positive_inter = int(len(Ip))
  
  for train_index, test_index in group_kfold.split(range(nb_positive_inter), groups=groups):
      # 9' pour train

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      ##### TEST ####
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
      print("nb of interactions + in test",len(indice_P_t))
      I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[1]==x[1]]
      print("number of interactions + deleted in test", len(set(I_t)))
      indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
      print("number of interactions + in test", len(indice_P_t))

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
      print("number of interactions -", a.shape)
      indice_N_t = np.array([-1, -1]).reshape(1,2)

      S_a = np.random.permutation(len(a))
      for i in S_a:
            if  (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
                indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
            if len(indice_N_t) == indice_P_t.shape[0] + 1:
                i_end_a = i
                print("i_end", i_end_a)
                break
        
      # add interactions np.nan in test
      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
      print("number of np.nan", c.shape)
      S_c = np.random.permutation(len(c))

      for i in S_c:
            if (c[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
                indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
            if len(indice_N_t) == indice_P_t.shape[0] + 1:
                i_end_c = i
                print("i_end", i_end_c)
                break

      indice_N_t = indice_N_t[:len(indice_P_t),:] #
      print("number of interactions - in test",len(indice_N_t))
      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]
      print("test", test.shape)
      all_test_interactions_arr.append(test)


  train_arr = []
  for i in range(len(all_train_interactions_arr)):
    df_train = pd.DataFrame(all_train_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_train_S = df_train.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_train_S = df_train_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_train_S = df_train_S[["smiles","fasta","label"]]
    df_train_S.columns = ["SMILES","Target Sequence","Label"]
    train_arr.append(df_train_S)

  with open(path_mkdir+"/train_arr.pkl","wb") as f:
    pickle.dump(train_arr,f)

  test_arr = []
  for i in range(len(all_test_interactions_arr)):
    df_test = pd.DataFrame(all_test_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_test_S = df_test.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_test_S = df_test_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_test_S = df_test_S[["smiles","fasta","label"]]
    df_test_S.columns = ["SMILES","Target Sequence","Label"]
    test_arr.append(df_test_S)

  with open(path_mkdir+"/test_arr.pkl","wb") as f:
    pickle.dump(test_arr,f)

  print("Train/test datasets prepared.")
  return train_arr, test_arr

def make_CV_train_test_unseen_target(df,nb_folds,path_mkdir):
  """
    Splits the input DataFrame into cross-validation training and testing datasets, ensuring that the proteins in 
    the test set are not present in the train set. The function performs K-fold cross-validation with the specified 
    number of folds.

    :param df: Input data containing the columns 'indfasta', 'indsmiles', and 'score'.
    :type df: pandas.DataFrame
    :param nb_folds: Number of folds for cross-validation.
    :type nb_folds: int
    :param path_mkdir: Path to the directory where the output train and test sets will be saved as pickle files.
    :type path_mkdir: str
    :return: A tuple containing lists of DataFrames representing the training and testing datasets for each fold.
    :rtype: (list[pandas.DataFrame], list[pandas.DataFrame])

    Note:
    The function first converts the DataFrame into a numpy matrix where rows correspond to 'indfasta' (protein indices), 
    columns to 'indsmiles' (drug indices), and cell values to 'score' (interaction score). Positive (interaction score = 1) 
    and negative (interaction score = 0) interactions are identified and distributed into the folds. NaN values in 'score' 
    are treated as unknown interactions and handled separately. The folds are created such that the proteins in the test 
    set are not present in the train set. The function returns the training and testing datasets for each fold as lists of
    DataFrames.
  """

  try : 
    intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
  except:
    intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

  # Set the different folds

  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)

  groups = np.array(Ip) #folds are made on proteins
  group_kfold = GroupKFold(n_splits=nb_folds)

  nb_positive_inter = int(len(Ip)) 
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  for train_index, test_index in group_kfold.split(range(nb_positive_inter), groups=groups):
      # 9' pour train

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      ##### TEST ####
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
      print("nb of interactions + in test",len(indice_P_t))
      I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[0]==x[0]]
      print("number of interactions + deleted in test", len(set(I_t)))
      indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
      print("number of interactions + in test", len(indice_P_t))

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
      print("number of interactions -", a.shape)
      indice_N_t = np.array([-1, -1]).reshape(1,2)

      S_a = np.random.permutation(len(a))
      for i in S_a:
        if (a[i,0] not in train[:,0]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_a = i
            print("i_end", i_end_a)
            break
    
      # add interactions np.nan in test
      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
      print("number of np.nan", c.shape)
      S_c = np.random.permutation(len(c))

      for i in S_c:
        if (c[i,0] not in train[:,0]): # we drop the interactions- in train and the prot in train
            indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
        if len(indice_N_t) == indice_P_t.shape[0] + 1:
            i_end_c = i
            print("i_end", i_end_c)
            break

      indice_N_t = indice_N_t[:len(indice_P_t),:] #
      print("number of interactions - in test",len(indice_N_t))
      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]
      print("test", test.shape)
      all_test_interactions_arr.append(test)

  train_arr = []
  for i in range(len(all_train_interactions_arr)):
    df_train = pd.DataFrame(all_train_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_train_S = df_train.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_train_S = df_train_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_train_S = df_train_S[["smiles","fasta","label"]]
    df_train_S.columns = ["SMILES","Target Sequence","Label"]
    train_arr.append(df_train_S)

  with open(path_mkdir+"train_arr.pkl","wb") as f:
    pickle.dump(train_arr,f)

  test_arr = []
  for i in range(len(all_test_interactions_arr)):
    df_test = pd.DataFrame(all_test_interactions_arr[i],columns=['indfasta','indsmiles','label'])
    df_test_S = df_test.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
    df_test_S = df_test_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
    df_test_S = df_test_S[["smiles","fasta","label"]]
    df_test_S.columns = ["SMILES","Target Sequence","Label"]
    test_arr.append(df_test_S)

  with open(path_mkdir+"test_arr.pkl","wb") as f:
    pickle.dump(test_arr,f)

  print("Train/test datasets prepared.")
  return train_arr, test_arr

def make_CV_train_test_Orphan(df,nb_folds,path_mkdir):
    """
    Splits the input DataFrame into cross-validation training and testing datasets, ensuring that the proteins 
    and molecules in the test set are not present in the train set. The function performs K-fold cross-validation 
    with the specified number of folds.

    :param df: Input data containing the columns 'indfasta', 'indsmiles', and 'score'.
    :type df: pandas.DataFrame
    :param nb_folds: Number of folds for cross-validation.
    :type nb_folds: int
    :param path_mkdir: Path to the directory where the output train and test sets will be saved as pickle files.
    :type path_mkdir: str
    :return: A tuple containing lists of DataFrames representing the training and testing datasets for each fold.
    :rtype: (list[pandas.DataFrame], list[pandas.DataFrame])

    Note:
    The function first converts the DataFrame into a numpy matrix where rows correspond to 'indfasta' (protein indices), 
    columns to 'indsmiles' (drug indices), and cell values to 'score' (interaction score). Positive (interaction score = 1) 
    and negative (interaction score = 0) interactions are identified and distributed into the folds. NaN values in 'score' 
    are treated as unknown interactions and handled separately. The folds are created such that the proteins and molecules 
    in the test set are not present in the train set.
    """
    try : 
        intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    except:
        intMat = df.pivot_table(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    n_p,n_m = intMat.shape
    Ip, Jm = np.where(intMat==1)
    Inp, Jnm = np.where(intMat==0)
    Inkp, Jnkm = np.where(np.isnan(intMat))

    nP = df[["indfasta"]].drop_duplicates().reset_index().shape[0]
    nM = df[["indsmiles"]].drop_duplicates().reset_index().shape[0]

    all_train_interactions_arr = []
    all_test_interactions_arr = []

    for i in range(nb_folds):

        SP = np.random.permutation(nP)
        SM = np.random.permutation(nM)

        groups = []
        for ip,im in zip(Ip,Jm):
            if (ip in SP[:int(0.6*nP)]) and (im in SM[:int(0.7*nM)]):
                groups.append(0)
            elif (ip in SP[int(0.6*nP):]) and (im in SM[int(0.7*nM):]):
                groups.append(1)
            elif (ip in SP[:int(0.6*nP)]) and (im in SM[int(0.7*nM):]):
                groups.append(2)
            else:
                groups.append(3)
        # je voudrais faire qu'avec (1 dans le train) et (2 dans le test) mais je ne sais pas comment faire
        train_index = np.where(np.array(groups)==0)[0]
        test_index = np.where(np.array(groups)==1)[0]

        #Inp, Jnm = np.where(intMat==0)
        #Inkp, Jnkm = np.where(np.isnan(intMat))

        Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

        Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

        train = np.zeros([1,3], dtype=int)

        nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
        for i in range(nb_prot):

            j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

            indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
            indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
            indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

            indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
            indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

            indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
            indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

            nb_positive_interactions = len(indice_P)
            nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

            indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
            indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
            indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
            indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

            if len(indice_P) <= len(indice_freq_one_prot):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                    len(indice_P), replace = False)
            elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
                indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_N_one_prot_poss))
            elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
                indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_poss_one_prot, indice_N_one_prot_poss))
            else:
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
                #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
                #print(indice_poss_one_prot_NK.shape)
                indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

            Mp[indice_N_one_prot.astype(int)]-=1

            # this protein has been processed
            Mm[j] = 0

            indice = np.r_[indice_P,indice_N_one_prot].astype(int)
            etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
            A = np.stack((indice, etiquette), axis=-1)
            B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
            train = np.concatenate((train,B))

        train = train[1:]
        #with open('data/train_prot_mol_orphan.data', 'wb') as f:
        #    pickle.dump(train, f)
        all_train_interactions_arr.append(train)
        print("train", train.shape)

        ##### TEST ####
        # interactions + in test
        indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)
        print("nb of interactions + in test",len(indice_P_t))
        I_t = [i for i,elt in enumerate(indice_P_t) for x in train if elt[0]==x[0] or elt[1]==x[1]]
        print("number of interactions + deleted in test", len(set(I_t)))
        indice_P_t = np.delete(indice_P_t, list(set(I_t)) ,axis = 0)
        print("number of interactions + in test", len(indice_P_t))

        # interactions - in test
        a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix
        print("number of interactions -", a.shape)
        indice_N_t = np.array([-1, -1]).reshape(1,2)

        S_a = np.random.permutation(len(a))
        for i in S_a:
            if (a[i,0] not in train[:,0]) and (a[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
                indice_N_t = np.concatenate((indice_N_t, a[i].reshape(1,2)))
            if len(indice_N_t) == indice_P_t.shape[0] + 1:
                i_end_a = i
                print("i_end", i_end_a)
                break
        
        # add interactions np.nan in test
        c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix
        print("number of np.nan", c.shape)
        S_c = np.random.permutation(len(c))

        for i in S_c:
            if (c[i,0] not in train[:,0]) and (c[i,1] not in train[:,1]): # we drop the interactions- in train and the prot in train
                indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
            if len(indice_N_t) == indice_P_t.shape[0] + 1:
                i_end_c = i
                print("i_end", i_end_c)
                break

        indice_N_t = indice_N_t[:len(indice_P_t),:] #
        print("number of interactions - in test",len(indice_N_t))
        # we add the column of 0 for the etiquette
        indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
        test = np.r_[indice_P_t,indice_N_t]
        print("test", test.shape)
        all_test_interactions_arr.append(test)
        
    train_arr = []
    for i in range(len(all_train_interactions_arr)):
        df_train = pd.DataFrame(all_train_interactions_arr[i],columns=['indfasta','indsmiles','label'])
        df_train_S = df_train.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
        df_train_S = df_train_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
        df_train_S = df_train_S[["smiles","fasta","label"]]
        df_train_S.columns = ["SMILES","Target Sequence","Label"]
        train_arr.append(df_train_S)

    with open(path_mkdir+"train_arr.pkl","wb") as f:
        pickle.dump(train_arr,f)

    test_arr = []
    for i in range(len(all_test_interactions_arr)):
        df_test = pd.DataFrame(all_test_interactions_arr[i],columns=['indfasta','indsmiles','label'])
        df_test_S = df_test.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
        df_test_S = df_test_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
        df_test_S = df_test_S[["smiles","fasta","label"]]
        df_test_S.columns = ["SMILES","Target Sequence","Label"]
        test_arr.append(df_test_S)

    with open(path_mkdir+"test_arr.pkl","wb") as f:
        pickle.dump(test_arr,f)

    print("Train/test datasets prepared.")
    return train_arr, test_arr

def make_CV_train_test(load_data,S,save_data,nb_folds=5):
    """
    Loads the interaction data from a CSV file, preprocesses the data to generate numerical indices for unique 
    smiles (molecules) and fasta (proteins), and splits the data into cross-validation training and testing datasets 
    based on the specified split type.

    :param load_data: Path to the input CSV file containing the interaction data : LCIdb_v2.csv (or after process_LCIdb with other thresholds) or Chembl.csv
    :type load_data: str
    :param S: Specifies the split type, which determines how the training and testing datasets are created. 
              Options are "full" (S1), "unseen_drug" (S2), "unseen_target" (S3), and "Orphan" (S4).
    :type S: str
    :param save_data: Path to the directory where the output train and test sets will be saved.
    :type save_data: str
    :param nb_folds: Number of folds for cross-validation, defaults to 5.
    :type nb_folds: int, optional
    :return: A tuple containing lists of DataFrames representing the training and testing datasets for each fold.

    Note:
    The function preprocesses the input data to create unique numerical indices for each unique molecule and 
    protein, which are then used to create a pivot table representing the interaction matrix. The data is split 
    into training and testing sets using the appropriate split function based on the specified split type (full, unseen_drug, 
    unseen_target, Orphan)
    """

    df = pd.read_csv(load_data,low_memory=False)
    try:
        df.rename(columns={'standardized smiles':'smiles'}, inplace=True)
    except:
        pass
    df_p = df[(df['score'] == 1)]
    df2 = df_p

    # give number to all smiles we keep and all fasta we keep (ie int +)
    # make dict smiles2ind and dict ind2smiles
    df_sm = df2[["smiles"]].drop_duplicates().reset_index()
    #df_sm = df_p[["standardized smiles"]].drop_duplicates().reset_index()
    df_sm.drop(columns=["index"],inplace=True)
    dict_ind2smiles = df_sm.to_dict()["smiles"]
    #dict_ind2smiles = df_sm.to_dict()["standardized smiles"]
    print("nombre de smiles: ",len(dict_ind2smiles))
    dict_smiles2ind = {v: k for k, v in dict_ind2smiles.items()}

    df_prot = df2[["fasta"]].drop_duplicates().reset_index()
    df_prot.drop(columns=["index"],inplace=True)
    dict_ind2fasta = df_prot.to_dict()["fasta"]
    print("nombre de fasta: ",len(dict_ind2fasta))
    dict_fasta2ind = {v: k for k, v in dict_ind2fasta.items()}

    # add this number to df
    #df["indsmiles"] = df["standardized smiles"].map(dict_smiles2ind)
    df["indsmiles"] = df["smiles"].map(dict_smiles2ind)
    df["indfasta"] = df["fasta"].map(dict_fasta2ind)

    # we drop when indsmiles is Nan
    indsmiles_index_with_nan = df.index[df.loc[:,"indsmiles"].isnull()]
    df = df.drop(indsmiles_index_with_nan,0)
    # we drop when indfasta is Nan
    indfasta_index_with_nan = df.index[df.loc[:,"indfasta"].isnull()]
    df = df.drop(indfasta_index_with_nan,0)

    intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    print("matrice d'interactions: ",intMat.shape)
    
    if S == "full":
        if not os.path.exists(save_data+"/full_data"):
            os.makedirs(save_data+"/full_data")
        all_train_interactions_arr, all_test_interactions_arr = make_CV_train_test_full(df,nb_folds,save_data+"/full_data")
    elif S == "unseen_drug":
        if not os.path.exists(save_data+"/unseen_drug"):
            os.makedirs(save_data+"/unseen_drug")
        all_train_interactions_arr, all_test_interactions_arr = make_CV_train_test_unseen_drug(df,nb_folds,save_data+"/unseen_drug")
        
    elif S == "unseen_target":
        if not os.path.exists(save_data+"/unseen_target"):
            os.makedirs(save_data+"/unseen_target")
        all_train_interactions_arr, all_test_interactions_arr = make_CV_train_test_unseen_target(df,nb_folds,save_data+"/unseen_target")
    elif S == "Orphan":
        if not os.path.exists(save_data+"/Orphan"):
            os.makedirs(save_data+"/Orphan")
        all_train_interactions_arr, all_test_interactions_arr = make_CV_train_test_Orphan(df,nb_folds,save_data+"Orphan")    
    print("Train datasets prepared.")
    return all_train_interactions_arr, all_test_interactions_arr

def make_trains_full(load_data,save_data,nb_trains=5):
    """
    Loads the interaction data from a CSV file, preprocesses the data to generate numerical indices for unique 
    smiles (molecules) and fasta (proteins), and make nb_train training datasets on full data by chosing different negative interactions each time.

    :param load_data: Path to the input CSV file containing the interaction data : LCIdb_v2.csv (or after process_LCIdb with other thresholds) or Chembl.csv
    :type load_data: str
    :param save_data: Path to the directory where the output train sets will be saved.
    :type save_data: str
    :param nb_trains: Number of different trains, defaults to 5.
    :type nb_trains: int, optional
    :return: A tuple containing lists of DataFrames representing the training datasets for each fold.

    Note:
    The function preprocesses the input data to create unique numerical indices for each unique molecule and 
    protein, which are then used to create a pivot table representing the interaction matrix. 
    Choose different negative interactions each time.
    """

    df = pd.read_csv(load_data,low_memory=False)
    try:
        df.rename(columns={'standardized smiles':'smiles'}, inplace=True)
    except:
        pass
    df_p = df[(df['score'] == 1)]
    df2 = df_p

    # give number to all smiles we keep and all fasta we keep (ie int +)
    # make dict smiles2ind and dict ind2smiles
    df_sm = df2[["smiles"]].drop_duplicates().reset_index()
    #df_sm = df_p[["standardized smiles"]].drop_duplicates().reset_index()
    df_sm.drop(columns=["index"],inplace=True)
    dict_ind2smiles = df_sm.to_dict()["smiles"]
    #dict_ind2smiles = df_sm.to_dict()["standardized smiles"]
    print("nombre de smiles: ",len(dict_ind2smiles))
    dict_smiles2ind = {v: k for k, v in dict_ind2smiles.items()}

    df_prot = df2[["fasta"]].drop_duplicates().reset_index()
    df_prot.drop(columns=["index"],inplace=True)
    dict_ind2fasta = df_prot.to_dict()["fasta"]
    print("nombre de fasta: ",len(dict_ind2fasta))
    dict_fasta2ind = {v: k for k, v in dict_ind2fasta.items()}

    # add this number to df
    #df["indsmiles"] = df["standardized smiles"].map(dict_smiles2ind)
    df["indsmiles"] = df["smiles"].map(dict_smiles2ind)
    df["indfasta"] = df["fasta"].map(dict_fasta2ind)

    # we drop when indsmiles is Nan
    indsmiles_index_with_nan = df.index[df.loc[:,"indsmiles"].isnull()]
    df = df.drop(indsmiles_index_with_nan,0)
    # we drop when indfasta is Nan
    indfasta_index_with_nan = df.index[df.loc[:,"indfasta"].isnull()]
    df = df.drop(indfasta_index_with_nan,0)

    intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)
    print("matrice d'interactions: ",intMat.shape)

    n_p,n_m = intMat.shape
    Ip, Jm = np.where(intMat==1)
    nb_positive_inter = int(len(Ip))
    Inp, Jnm = np.where(intMat==0)
    Inkp, Jnkm = np.where(np.isnan(intMat))
    
    all_train_interactions_arr = []
    for i_train in range(nb_trains):
        Mm, bin_edges = np.histogram(df["indfasta"], bins = range(intMat.shape[0]+1))
        Mp, bin_edges = np.histogram(df["indsmiles"], bins = range(intMat.shape[1]+1))
        train = np.zeros([1,3], dtype=int)

        nb_prot = len(list(set(Ip))) # number of different prot in train
        for i in range(nb_prot):

            j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

            indice_P = Jm[np.where(Ip==j)[0]]  #np.array with index of interactions + in train
            indice_N = [k for k in Jm if intMat[j][k]==0]
            indice_NK = [k for k in Jm if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

            indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
            indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

            indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
            indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

            nb_positive_interactions = len(indice_P)
            nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

            indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
            indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
            indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
            indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

            if len(indice_P) <= len(indice_freq_one_prot):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                    len(indice_P), replace = False)
            elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
                indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_N_one_prot_poss))
            elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
                indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_poss_one_prot, indice_N_one_prot_poss))
            else:
                # we shoot at random nb_positive_interactions in drugs with a lot of interactions
                nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
                #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
                #print(indice_poss_one_prot_NK.shape)
                indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                        nb_negative_interactions_remaining, replace = False )
                indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                                indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

            Mp[indice_N_one_prot.astype(int)]-=1

            # this protein has been processed
            Mm[j] = 0

            indice = np.r_[indice_P,indice_N_one_prot].astype(int)
            etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
            A = np.stack((indice, etiquette), axis=-1)
            B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
            train = np.concatenate((train,B))

        train = train[1:]
        all_train_interactions_arr.append(train)
        print("train", train.shape)

    train_arr = []
    for i in range(len(all_train_interactions_arr)):
        df_train = pd.DataFrame(all_train_interactions_arr[i],columns=['indfasta','indsmiles','label'])
        df_train_S = df_train.merge(df[["indfasta","fasta"]].drop_duplicates(),on="indfasta")
        df_train_S = df_train_S.merge(df[["indsmiles","smiles"]].drop_duplicates(),on="indsmiles")
        df_train_S = df_train_S[["smiles","fasta","label"]]
        df_train_S.columns = ["SMILES","Target Sequence","Label"]
        train_arr.append(df_train_S)

    with open(save_data+"/train_arr.pkl","wb") as f:
        pickle.dump(train_arr,f)

    return train_arr

def process_LCIdb(name_file, data_dir = "./", max_length_fasta = 1000, bioactivity_choice = "checkand1database",min_weight = 100, max_weight = 900,  interaction_plus = 1e-7, interaction_minus = 1e-4):
    """
    Processes data from a given ligand-chemical interaction database file and performs various data cleaning and transformation steps.

    :param name_file: The name of the file containing the database information.
    :type name_file: str
    :param data_dir: The directory where the database file and the file prot_uniprot_fasta_all.csv are located, defaults to "./".
    :type data_dir: str, optional
    :param max_length_fasta: The maximum length of FASTA sequences to be included, defaults to 1000.
    :type max_length_fasta: int, optional
    :param bioactivity_choice: The type of bioactivity data to include, defaults to 'checkand1database'.
    :type bioactivity_choice: str, optional
    :param min_weight: The minimum molecular weight of the ligands to be included, defaults to 100.
    :type min_weight: int, optional
    :param max_weight: The maximum molecular weight of the ligands to be included, defaults to 900.
    :type max_weight: int, optional
    :param interaction_plus: The threshold for positive interaction, defaults to 1e-7.
    :type interaction_plus: float, optional
    :param interaction_minus: The threshold for negative interaction, defaults to 1e-4.
    :type interaction_minus: float, optional

    :returns: A pandas DataFrame containing the processed and filtered data.
    :rtype: pandas.DataFrame

    The function performs the following steps:
    - Reads the database and filters based on activity type (pIC50, pKd, pKi) and units (negative logarithm).
    - Merges with protein FASTA sequence data and filters based on sequence length and molecular weight.
    - Processes bioactivity data based on the selected type and calculates mean, min, and max bioactivity values.
    - Aggregates data based on SMILES representation and FASTA sequence, and calculates interaction scores.
    - Saves the processed data to a new CSV file.
    """

    # download
    df = pd.read_csv(data_dir+name_file,low_memory=False)
    print("all")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# targets",df["Target"].drop_duplicates().shape)

    # keep only pIC50, pKd and pKi
    df = df[((df["Activity type"]=='pIC50') | (df["Activity type"]=='pKd')) | (df["Activity type"]=='pKi')   ]#.dropna(axis = 1, how = 'all')
    print("pIC50, pKd and pKi")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# targets",df["Target"].drop_duplicates().shape)

    # keep only ligns which the negative logarithm of the measure is known
    df = df[df["Unit"]=='neg. log']#.dropna(axis = 1, how = 'all')
    print("Unit neg. log")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# targets",df["Target"].drop_duplicates().shape)

    # add uniprot,fasta for human proteins
    df_hgnc_symbol_check = pd.read_csv(data_dir +'prot_uniprot_fasta_all.csv')
    df = pd.merge(df, df_hgnc_symbol_check)
    print("add uniprot,fasta")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    #keep proteins with fasta length < max_length_fasta
    df = df[df['fasta']< max_length_fasta*'Z']
    print("fasta length < "+str(max_length_fasta))
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    #bioactivity values;
    if bioactivity_choice == 'check':
            df = df[df["Activity check annotation"]==' ']
    elif bioactivity_choice == 'check_and_1database':
        df = df[(df["Activity check annotation"]==' ')| (df["Activity check annotation"]=='only 1 data point')]
    print("Activity check annotation")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# targets",df["Target"].drop_duplicates().shape)

    # keep only ligns which the Smiles is ckecked
    df = df[ (df["Structure check (Tanimoto)"]=='match') |(df["Structure check (Tanimoto)"]=='1 structure') |(df["Structure check (Tanimoto)"]=='no match (1)') ]#.dropna(axis = 1, how = 'all')
    print("Structure check (Tanimoto)")
    print("# Ligand names",df["Ligand names"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    # put index from 0 to len(df)
    df = df.reset_index(drop=True)

    # we keep one smiles for each lign
    def f(i):
        p = df.shape[1]
        for j in range(p-13,p-8):
            if isinstance(df.iloc[i,j],str) :
                return df.iloc[i,j]

    df["smiles"] = list(map(lambda i: f(i), df.index))

    # keep molecule with mol weight (µM) between 100 and 900
    liste = []
    for clé in df[["smiles"]].values:
        try:
            m = Chem.MolFromSmiles(clé[0]);
            mol_weight = D.MolWt(m);
            liste.append(mol_weight);
        except :
            mol_weight = 0
            liste.append(mol_weight)

    df['mol_weight'] = pd.Series(liste, index = df.index)
    df = df[df['mol_weight'] <= max_weight]
    df = df[df['mol_weight'] >= min_weight]
    print(f"mol_weight between {min_weight} and {max_weight}")
    print("# smiles",df["smiles"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    # put index from 0 to len(df)
    df = df.reset_index(drop=True)

    # calculate the mean,min,max of the bioactivity
    def catch(func, handle=lambda e : e, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return np.nan

    def f(i):
        p = df.shape[1]
        l_m = [catch(lambda:float(df.iloc[i,j].split('*')[0])) for j in range(7,p-18)]
        # quit np.nan of lm
        l_m = [l_m[j] for j in range(len(l_m)) if not(l_m[j] is np.nan)]

        l_p = [catch(lambda:float(df.iloc[i,j].split('*')[1][1:-1])) for j in range(7,p-18)]
        # quit np.nan of lp
        l_p = [l_p[j] for j in range(len(l_p)) if not(l_p[j] is np.nan)]

        # calculate the ponderated mean of the bioactivity
        m = sum([l_m[j]*l_p[j] for j in range(len(l_m))])/sum(l_p)
        return m,np.nanmin(l_m),np.nanmax(l_m)

    df["mean,min,max"] = list(map(lambda i: f(i), df.index))

    df["mean"] = list(map(lambda i:float(df.iloc[i,-1][0]), df.index))
    df["min"] = list(map(lambda i:float(df.iloc[i,-2][1]), df.index))
    df["max"] = list(map(lambda i: float(df.iloc[i,-3][2]), df.index))

    # find column with (Ligand names,Target) and diferrent activity type (pKd, pKi, PIC50) and mean
    aggregation_functions = {"ChEMBL ID" : 'first',"PubChem ID" : 'first',"IUPHAR ID" : 'first', "Ligand names":"first", "Target":'first',"uniprot":'first', "mean": 'mean',"min": 'min',"max": 'max'}
    df = df.groupby(["smiles",'fasta',"Activity type"]).aggregate(aggregation_functions)
    df = df.reset_index()  #pour enlever multi-index
    print("agregation mean, min, max by smiles,fasta, activity type")
    print("# smiles",df["smiles"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    # make a column pIC50 with the 'mean' if "Activity type" == pIC50
    df["mean pIC50"] = df["mean"]*(df["Activity type"]=='pIC50')
    df["mean pKi"] = df["mean"]*(df["Activity type"]=='pKi')
    df["mean pKd"] = df["mean"]*(df["Activity type"]=='pKd')

    # replace 0 in  pIC50, pKi, pKd by nan
    df["mean pIC50"] = df["mean pIC50"].replace(0, np.nan)
    df["mean pKi"] = df["mean pKi"].replace(0, np.nan)
    df["mean pKd"] = df["mean pKd"].replace(0, np.nan)

    # make a column pIC50 with the 'min' if "Activity type" == pIC50
    df["min pIC50"] = df["min"]*(df["Activity type"]=='pIC50')
    df["min pKi"] = df["min"]*(df["Activity type"]=='pKi')
    df["min pKd"] = df["min"]*(df["Activity type"]=='pKd')

    # replace 0 in  pIC50, pKi, pKd by nan
    df["min pIC50"] = df["min pIC50"].replace(0, np.nan)
    df["min pKi"] = df["min pKi"].replace(0, np.nan)
    df["min pKd"] = df["min pKd"].replace(0, np.nan)

    # make a column pIC50 with the 'max' if "Activity type" == pIC50
    df["max pIC50"] = df["max"]*(df["Activity type"]=='pIC50')
    df["max pKi"] = df["max"]*(df["Activity type"]=='pKi')
    df["max pKd"] = df["max"]*(df["Activity type"]=='pKd')

    # replace 0 in  pIC50, pKi, pKd by nan
    df["max pIC50"] = df["max pIC50"].replace(0, np.nan)
    df["max pKi"] = df["max pKi"].replace(0, np.nan)
    df["max pKd"] = df["max pKd"].replace(0, np.nan)

    # find column with (Ligand names,Target) and diferrent activity type (pKd, pKi, PIC50) and mean
    aggregation_functions = {"ChEMBL ID" : 'first',"PubChem ID" : 'first',"IUPHAR ID" : 'first', "Ligand names":"first", "Target":'first',"uniprot":'first', 
                            "mean": 'mean',"mean pIC50":"mean","mean pKi":"mean","mean pKd":"mean", "min": 'min',"min pIC50":"min","min pKi":"min","min pKd":"min","max": 'max',"max pIC50":"max","max pKi":"max","max pKd":"max"}
    df = df.groupby(["smiles",'fasta']).aggregate(aggregation_functions)
    df = df.reset_index()  #pour enlever multi-index
    print("agregation mean, min, max by smiles,fasta")
    print("# smiles",df["smiles"].drop_duplicates().shape)
    print("# fasta",df["fasta"].drop_duplicates().shape)

    score_plus = -np.log10(interaction_plus)
    score_minus = -np.log10(interaction_minus)


    def calcul_score_mean(x):
        if x >= score_plus:
            return 1
        elif x > score_minus:
            return 0.5
        else:
            return 0
        
    def calcul_score_mean_2(i,x):
        m = "min "+x
        M = "max "+x
        moy = "mean "+x
        if (df.loc[i,moy] >= score_plus):
            if (df.loc[i,M] - df.loc[i,m])<= 1:
                return 1
            elif df.loc[i,m] >= score_plus:
                return 1
            else:
                return 0.5
        elif df.loc[i,moy] <= score_minus:
            if (df.loc[i,M] - df.loc[i,m])<= 1:
                return 0
            elif df.loc[i,M] <= score_minus:
                return 0
            else:
                return 0.5 
        else:
            return 0.5


    def calcul_score_minMax(i,x):
        m = "min "+x
        M = "max "+x
        if df.loc[i,m] >= score_plus:
            return 1
        elif df.loc[i,M] <= score_minus:
            return 0
        else:
            return 0.5
            
    def f_score_2(i):
        # If a given compound and protein target have multiple measurements of different types,
        # we choose them in the following order of preference: Kd over Ki over IC50.
        if not np.isnan(df.loc[i,"mean pKd"]):
            return calcul_score_mean_2(i,"pKd")
        elif not np.isnan(df.loc[i,"mean pKi"]):
            return calcul_score_mean_2(i,"pKi")
        elif not np.isnan(df.loc[i,"mean pIC50"]):
            return calcul_score_mean_2(i,"pIC50")
        else:
            if not np.isnan(df.loc[i,"min pKd"]):
                return calcul_score_minMax(i,"pKd")
            elif not np.isnan(df.loc[i,"min pKi"]):
                return calcul_score_minMax(i,"pKi")
            elif not np.isnan(df.loc[i,"min pIC50"]):
                return calcul_score_minMax(i,"pIC50")
            else:
                return np.nan

    df['score'] = list(map(lambda i: f_score_2(i), df.index))
    print("score calculé")
    print("df.shape = ",df.shape)

    # register the dataframe
    df.to_csv(data_dir+"LCIdb.csv", index = False)

    return df


def MT_komet(data_set,lambda_list,mM,dM):
    """
    To compair a single-task NN SVM approach to the multi-task Komet algorithm in several settings. 
    For each protein in LCIdb considered in turn as the query protein, we performed the following experiment: a test set was built, comprising all known positive DTIs 
    involving the query protein in LCIdb and their balanced negative DTIs. A corresponding training set is built: For Komet, it consists of all DTIs remaining in LCIdb after removal of DTIs 
    that are in the test set, so that the query protein is orphan. The AUPR is calculated on the test set. 
    The process is repeated for each protein in LCIdb, and the average AUPR is calculated.

    :param data_set: Dataframe containing the dataset (train) with columns 'SMILES', 'Target Sequence', and 'Label'.
    :type data_set: pandas.DataFrame
    :param lambda_list: List of regularization parameters to use for training.
    :type lambda_list: list[float]
    :param mM: Number of molecules to use for the Nystrom approximation.
    :type mM: int
    :param dM: Final dimension of features for molecules.
    :type dM: int
    :return: DataFrame with the evaluation metrics for each protein.
    :rtype: pandas.DataFrame

    Note:
    See section 7 of Supporting Information.
    """
    data_dir = './data/'

    # Load Protein kernel and dictionary of index
    dict_ind2fasta_all = pickle.load(open(data_dir + "dict_ind2fasta_all.data", 'rb'))
    dict_fasta2ind_all = {fasta:ind for ind,fasta in dict_ind2fasta_all.items()}
    with open(data_dir + "dict_ind2fasta_all_K_prot.data", 'rb') as f:
        KP_all = pickle.load(f)
    KP_all.shape, type(KP_all)
    
    full = data_set
       
    #### MOLECULE####
    list_smiles = full[['SMILES']].drop_duplicates().values.flatten()
    nM = len(list_smiles)
    # add indsmiles in train, val, test
    dict_smiles2ind = {list_smiles[i]:i for i in range(nM)}
    # molecule kernel_first step : compute Morgan FP for each smiles of all the dataset
    MorganFP = Morgan_FP(list_smiles)

    # In case there are less molecules than the number of molecules to compute the Nystrom approximation
    mM = min(mM,nM) # number of molecule to compute nystrom
    dM = min(dM,nM) # final dimension of features for molecules

    # compute the Nystrom approximation of the mol kernel and the features of the Kronecker kernel (features normalized and calculated on all mol contained in the dataset (train/val/test))
    X_cn = Nystrom_X_cn(mM,dM,nM,MorganFP)
    #print("mol features shape",X_cn.shape)

    #### PROTEIN####
    # Index of the protein in the dataset
    fasta = full[['Target Sequence']].drop_duplicates().values.flatten() # fasta sequence on the dataset, in the same order as the dataset
    I_fasta = [int(dict_fasta2ind_all[fasta[i]]) for i in range(len(fasta))] # index of fasta in the precomputed dict and protein kernel, in the same order as the dataset
    KP = KP_all[I_fasta,:][:,I_fasta]
    KP = torch.tensor(KP, dtype=mytype).to(device)
    print("kernel prot shape",KP.shape)

    # computation of feature for protein (no nystrom, just SVD)
    rP = KP.shape[0]#min(KP.shape[0],500)
    U, Lambda, VT = torch.svd(KP)
    Y = U[:,:rP] @ torch.diag(torch.sqrt(Lambda[:rP]))

    # nomramlisation of the features
    Y_c = Y - Y.mean(axis = 0)
    Y_cn = Y_c / torch.norm(Y_c,dim = 1)[:,None]
    print("protein features shape",Y.shape)

    # creation df vide avec colums columns=['fasta','nb_test','au_PR','au_Roc','acc'])
    df = pd.DataFrame(columns=['Target','family','nb_test_1','nb_test_0','au_PR','au_Roc','acc'])

    try : 
        df_family = pd.read_csv(data_dir+"fasta_uniprot_family.csv")
    except:
        print("No family file")

    for i in range(len(fasta)):
    #for i in range(debut,fin):
        l_info = []
        #l_info.append(fasta[i])
        l_info+= [df_family[df_family['fasta']==fasta[i]]['Target'].values[0],df_family[df_family['fasta']==fasta[i]]['family'].values[0]]

        #df_test : only DTI known in full
        df_test = full[full['Target Sequence'] == fasta[i]]
        l_info.append(len(df_test[df_test['Label']==1]))
        l_info.append(len(df_test[df_test['Label']==0]))

        #df_train: protein tested is orphan in full
        df_train = full[full['Target Sequence'] != fasta[i]]

        # we add indices in df_test and df_train
        df_test['indfasta'] = i
        df_train['indfasta'] = df_train['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])

        df_train['indsmiles'] = df_train['SMILES'].apply(lambda x:dict_smiles2ind[x] )
        df_test['indsmiles'] = df_test['SMILES'].apply(lambda x: dict_smiles2ind[x])

        # Train
        I, J, y = load_datas(df_train)
        n = len(I)
        #print("len(train)",n)

        # Test
        I_test, J_test, y_test = load_datas(df_test)
        n_test = len(I_test)
        #print("len(test)",n_test)

        #### TRAINING ####
        
        for j,lamb in enumerate(lambda_list):
                #print('lambda',lamb)
                w_bfgs,b_bfgs,h = SVM_bfgs(X_cn,Y_cn,y,I,J,lamb,niter=50)
                # we compute a probability using weights (Platt scaling)
                s,t,h2 = compute_proba_Platt_Scalling(w_bfgs,X_cn,Y_cn,y,I,J,niter=20)
                #### TEST ####
                # we compute a probability using weights (Platt scaling)
                m,y_pred, proba_pred = compute_proba(w_bfgs,b_bfgs,s,t,X_cn,Y_cn,I_test,J_test)
                # we compute the results
                acc1,au_Roc,au_PR,thred_optim,acc_best,cm,FP = results(y_test.cpu(),y_pred.cpu(),proba_pred.cpu())

        l_info+=[au_PR,au_Roc,acc1]
        # rajouter une ligne dans le dataframe
        df.loc[i] = l_info
    
    return df

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score

def NN_ST_SVM(data_set,mM,dM,lbda):
    """
    Trains and tests a Neirest Neighbour, single-task SVM model.
    For each protein, we performed the following experiment: a test set was built, comprising all known positive DTIs involving the query protein in LCIdb and their balanced negative DTIs. 
    A corresponding training set is built: the training set consists of all positive and negative DTIs involving the nearest protein, according to the LAkernel similarity. 
    We traina linear SVM using the same molecular features as Komet. The AUPR is calculated on the test set.

    :param data_set: Dataframe containing the dataset (train) with columns 'SMILES', 'Target Sequence', and 'Label'.
    :type data_set: pandas.DataFrame
    :param mM: Number of molecules to use for the Nystrom approximation.
    :type mM: int
    :param dM: Final dimension of features for molecules.
    :type dM: int
    :param lbda: Regularization parameter for the SVM model.
    :type lbda: float
    :return: DataFrame with the evaluation metrics for each protein.
    :rtype: pandas.DataFrame

    Note:
    Section 7 of Supporting Information of the paper.
    """
    
    data_dir = './data/'

    # Load Protein kernel and dictionary of index
    dict_ind2fasta_all = pickle.load(open(data_dir + "dict_ind2fasta_all.data", 'rb'))
    dict_fasta2ind_all = {fasta:ind for ind,fasta in dict_ind2fasta_all.items()}
    with open(data_dir + "dict_ind2fasta_all_K_prot.data", 'rb') as f:
        KP_all = pickle.load(f)
    KP_all.shape, type(KP_all)
    
    full= data_set
       
    #### MOLECULE####
    list_smiles = full[['SMILES']].drop_duplicates().values.flatten()
    nM = len(list_smiles)
    #print("number of different smiles (mol):",nM)

    # add indsmiles in train, val, test
    dict_smiles2ind = {list_smiles[i]:i for i in range(nM)}

    # molecule kernel_first step : compute Morgan FP for each smiles of all the dataset
    MorganFP = Morgan_FP(list_smiles)

    # In case there are less molecules than the number of molecules to compute the Nystrom approximation
    mM = min(mM,nM) # number of molecule to compute nystrom
    dM = min(dM,nM) # final dimension of features for molecules

    # compute the Nystrom approximation of the mol kernel and the features of the Kronecker kernel (features normalized and calculated on all mol contained in the dataset (train/val/test))
    X_cn = Nystrom_X_cn(mM,dM,nM,MorganFP)
    #print("mol features shape",X_cn.shape)

    #### PROTEIN####
    # Index of the protein in the dataset
    fasta = full[['Target Sequence']].drop_duplicates().values.flatten() # fasta sequence on the dataset, in the same order as the dataset
    #print("number of different Fasta (protein):",len(fasta))
    I_fasta = [int(dict_fasta2ind_all[fasta[i]]) for i in range(len(fasta))] # index of fasta in the precomputed dict and protein kernel, in the same order as the dataset
    KP = KP_all[I_fasta,:][:,I_fasta]
    KP = torch.tensor(KP, dtype=mytype).to(device)
    print("kernel prot shape",KP.shape)

    # computation of feature for protein (no nystrom, just SVD)
    rP = KP.shape[0]#min(KP.shape[0],500)
    U, Lambda, VT = torch.svd(KP)
    Y = U[:,:rP] @ torch.diag(torch.sqrt(Lambda[:rP]))

    # nomramlisation of the features
    Y_c = Y - Y.mean(axis = 0)
    Y_cn = Y_c / torch.norm(Y_c,dim = 1)[:,None]
    print("protein features shape",Y.shape)

    df = pd.DataFrame(columns=['Target','family','nb_test_0','nb_test_1','sim_K','nearest_prot_K','family_K','nb_train_0','nb_train_1','au_PR','au_Roc','acc','lbda'])
    try : 
        df_family = pd.read_csv(data_dir+"fasta_uniprot_family.csv")
    except:
        print("No family file")

    for i in range(len(fasta)):
        l_info = []
        l_info+= [df_family[df_family['fasta']==fasta[i]]['Target'].values[0],df_family[df_family['fasta']==fasta[i]]['family'].values[0]]

        #df_test : only DTI known in full
        df_test = full[full['Target Sequence'] == fasta[i]]
        l_info.append(len(df_test[df_test['Label']==1]))
        l_info.append(len(df_test[df_test['Label']==0]))

        # we search the nearest protein in the dataset
        sim = np.dot(Y_cn[i],Y_cn.T)
        I_prot = np.argsort(sim, kind='stable')[-2:]
        l_info.append(sim[I_prot[0]])
        l_info+= [df_family[df_family['fasta']==fasta[I_prot[0]]]['Target'].values[0],df_family[df_family['fasta']==fasta[I_prot[0]]]['family'].values[0]]
        
        df_train = full[full['Target Sequence'] == fasta[I_prot[0]]]
        l_info.append(len(df_train[df_train['Label']==1]))
        l_info.append(len(df_train[df_train['Label']==0]))

        # we add indices in df_test and df_train
        df_test['indfasta'] = i
        df_train['indfasta'] = I_prot[0]

        # we add indices in df_test and df_train
        df_test['indsmiles'] = df_test['SMILES'].apply(lambda x:dict_smiles2ind[x] )
        df_train['indsmiles'] = df_train['SMILES'].apply(lambda x:dict_smiles2ind[x] )

        X = X_cn[df_train["indsmiles"].values]
        y = df_train["Label"].values

        C_train = 1/(len(y)*lbda)
        clf = SVC(probability=True,C = C_train)  #hyperparameter to be tuned

        clf.fit(X, y)
        X_test = X_cn[df_test["indsmiles"].values].detach().numpy()
        y_test = df_test["Label"].values
        y_pred = clf.predict(X_test)
        #accuracy
        acc1 = accuracy_score(y_test, y_pred)
        y_proba = clf.predict_proba(X_test)
        # print(classification_report(y_test, y_pred))
        #auc
        fpr, tpr, thresholds = roc_curve(y_test,y_proba[:,1])
        auc_score = auc(fpr,tpr)
        #aupr
        average_precision = average_precision_score(y_test,y_proba[:,1])
            
        l_info+=[average_precision,auc_score,acc1]
        l_info.append(lbda)
        # rajouter une ligne dans le dataframe
        df.loc[i] = l_info
    return df

def predict_drug_profile(train,smiles_drug,mM = 3000,dM = 1000,lamb = 1e-6): 
    """
    Predicts the interaction profile of a given drug (specified by its SMILES representation) against a set of 
    proteins by training Komet 5 times. Each time, Komet uses different training datasets (different choices of negatives) and different molecular landmarks. 

    :param train: List of training datasets, each represented as a DataFrame.
    :type train: list[pandas.DataFrame]
    :param smiles_drug: SMILES representation of the drug to predict interactions for.
    :type smiles_drug: str
    :param mM: Number of molecules to use for the Nystrom approximation, defaults to 3000.
    :type mM: int, optional
    :param dM: Final dimension of features for molecules, defaults to 1000.
    :type dM: int, optional
    :param lamb: Regularization parameter for the SVM model, defaults to 1e-6.
    :type lamb: float, optional
    :return: A DataFrame with the predicted interaction profile for each protein.
    :rtype: pandas.DataFrame

    Note:
    This function expects precomputed protein kernels and dictionaries mapping protein sequences to indices.
    It uses Morgan fingerprints for molecule features and applies Nystrom approximation to the molecule kernel.
    The protein features are computed using SVD. The function trains an SVM model for each training dataset 
    and predicts the interaction probabilities for the specified drug.
    """   
    data_dir = './data/'
    # Load Protein kernel and dictionary of index
    dict_ind2fasta_all = pickle.load(open(data_dir + "dict_ind2fasta_all.data", 'rb'))
    dict_fasta2ind_all = {fasta:ind for ind,fasta in dict_ind2fasta_all.items()}
    with open(data_dir + "dict_ind2fasta_all_K_prot.data", 'rb') as f:
        KP_all = pickle.load(f)
    
    full = train[0]
       
    #### MOLECULE####
    list_smiles = full[['SMILES']].drop_duplicates().values.flatten()

    if smiles_drug not in list_smiles:
        print("The drug is not in the dataset")
        # add this drug in list_smiles
        list_smiles = np.append(list_smiles,smiles_drug)
    
    nM = len(list_smiles)
    dict_smiles2ind = {list_smiles[i]:i for i in range(nM)}
    # molecule kernel_first step : compute Morgan FP for each smiles of all the dataset
    MorganFP = Morgan_FP(list_smiles)

    # In case there are less molecules than the number of molecules to compute the Nystrom approximation
    mM = min(mM,nM) # number of molecule to compute nystrom
    dM = min(dM,nM) # final dimension of features for molecules

    #### PROTEIN####
    # Index of the protein in the dataset
    fasta = full[['Target Sequence']].drop_duplicates().values.flatten() # fasta sequence on the dataset, in the same order as the dataset
    I_fasta = [int(dict_fasta2ind_all[fasta[i]]) for i in range(len(fasta))] # index of fasta in the precomputed dict and protein kernel, in the same order as the dataset
    KP = KP_all[I_fasta,:][:,I_fasta]
    KP = torch.tensor(KP, dtype=mytype).to(device)
    print("kernel prot shape",KP.shape)

    # computation of feature for protein (no nystrom, just SVD)
    rP = KP.shape[0]#min(KP.shape[0],500)
    U, Lambda, VT = torch.svd(KP)
    Y = U[:,:rP] @ torch.diag(torch.sqrt(Lambda[:rP]))

    # nomramlisation of the features
    Y_c = Y - Y.mean(axis = 0)
    Y_cn = Y_c / torch.norm(Y_c,dim = 1)[:,None]
    print("protein features shape",Y.shape)

    # we test the drug on each protein
    df_test = pd.DataFrame(columns = ['SMILES','Target Sequence','Label','Proba_predicted_mean','Proba_predicted_std','Proba_predicted_min','Proba_predicted_max'])

    df_test['Target Sequence'] = fasta
    df_test['SMILES'] = smiles_drug
    df_test['Label'] = full.apply(lambda x:  x['Label'] if (x['SMILES'] == smiles_drug) & (x['Target Sequence'] in fasta) else -1 ,axis = 1)

    #pred = np.zeros((len(fasta), len(train)))
    pred = np.zeros((len(fasta), len(train)))

    for i in range(len(train)):
    #for i in range(0):

        # compute the Nystrom approximation of the mol kernel and the features of the Kronecker kernel (features normalized and calculated on all mol contained in the dataset (train/val/test))
        X_cn = Nystrom_X_cn(mM,dM,nM,MorganFP)
        #print("mol features shape",X_cn.shape)

        #### TRAINING SET ####
        df_train = train[i].copy()

        # we add indices in df_test and df_train
        df_test['indfasta'] = df_test['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])
        df_train['indfasta'] = df_train['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])

        df_train['indsmiles'] = df_train['SMILES'].apply(lambda x: dict_smiles2ind[x])
        df_test['indsmiles'] = df_test['SMILES'].apply(lambda x: dict_smiles2ind[x])

        # Train
        I, J, y = load_datas(df_train)
        n = len(I)
        #print("len(train)",n)

        # Test
        I_test, J_test, y_test = load_datas(df_test)
        n_test = len(I_test)
        #print("len(test)",n_test)

        #### TRAINING ####
        # we train the model
        w_bfgs,b_bfgs,h = SVM_bfgs(X_cn,Y_cn,y,I,J,lamb,niter=50)
        # we compute a probability using weights (Platt scaling)
        s,t,h2 = compute_proba_Platt_Scalling(w_bfgs,X_cn,Y_cn,y,I,J,niter=20)
        #### TEST ####
        # we compute a probability using weights (Platt scaling)
        m,y_pred, proba_pred = compute_proba(w_bfgs,b_bfgs,s,t,X_cn,Y_cn,I_test,J_test)
        # we compute the results
        #acc1,au_Roc,au_PR,thred_optim,acc_best,cm,FP = results(y_test.cpu(),y_pred.cpu(),proba_pred.cpu())

        pred[:,i] = proba_pred.cpu().numpy()

    df_test['Proba_predicted_mean'] = np.mean(pred,axis = 1)
    df_test['Proba_predicted_std'] = np.std(pred,axis = 1)
    df_test['Proba_predicted_min'] = np.min(pred,axis = 1)
    df_test['Proba_predicted_max'] = np.max(pred,axis = 1)

    # sort the dataframe by the predicted probability
    df_test = df_test.sort_values(by = 'Proba_predicted_mean',ascending = False)

    #change Label on Label_known
    df_test = df_test.rename(columns = {'Label':'Label_known'})

    # add protein name and uniprot
    try : 
        df_family = pd.read_csv(data_dir+"fasta_uniprot_family.csv")
    except:
        print("No family file")
    
    df_test = df_test.merge(df_family,how = 'left', left_on = 'Target Sequence', right_on = 'fasta')
    df_test = df_test[['Target','uniprot','family','Label_known','Proba_predicted_mean','Proba_predicted_std','Proba_predicted_min','Proba_predicted_max']]

    return df_test

def predict_protein_profile(train,fasta_protein,mM = 3000,dM = 1000,lamb = 1e-6):  
    """
    Predicts the interaction profile of a given protein (specified by its FASTA sequence) against a set of 
    drugs using the KOMET model. The function trains Komet 5 times on different training datasets and different
    molecular landmarks to predict the interaction profile for the specified protein.

    :param train: List of training datasets, each represented as a DataFrame.
    :type train: list[pandas.DataFrame]
    :param fasta_protein: FASTA sequence of the protein to predict interactions for.
    :type fasta_protein: str
    :param mM: Number of molecules to use for the Nystrom approximation, defaults to 3000.
    :type mM: int, optional
    :param dM: Final dimension of features for molecules, defaults to 1000.
    :type dM: int, optional
    :param lamb: Regularization parameter for the SVM model, defaults to 1e-6.
    :type lamb: float, optional
    :return: A DataFrame with the predicted interaction profile for each drug.
    :rtype: pandas.DataFrame

    Note:
    This function expects precomputed protein kernels and dictionaries mapping protein sequences to indices.
    It uses Morgan fingerprints for molecule features and applies Nystrom approximation to the molecule kernel.
    The protein features are computed using SVD. The function trains an SVM model for each training dataset 
    and predicts the interaction probabilities for the specified protein.
    """
      
    data_dir = './data/'
    # Load Protein kernel and dictionary of index
    dict_ind2fasta_all = pickle.load(open(data_dir + "dict_ind2fasta_all.data", 'rb'))
    dict_fasta2ind_all = {fasta:ind for ind,fasta in dict_ind2fasta_all.items()}
    with open(data_dir + "dict_ind2fasta_all_K_prot.data", 'rb') as f:
        KP_all = pickle.load(f)
    
    full = train[0]
       
    #### MOLECULE####
    list_smiles = full[['SMILES']].drop_duplicates().values.flatten()    
    nM = len(list_smiles)
    dict_smiles2ind = {list_smiles[i]:i for i in range(nM)}
    # molecule kernel_first step : compute Morgan FP for each smiles of all the dataset
    MorganFP = Morgan_FP(list_smiles)
    # In case there are less molecules than the number of molecules to compute the Nystrom approximation
    mM = min(mM,nM) # number of molecule to compute nystrom
    dM = min(dM,nM) # final dimension of features for molecules

    #### PROTEIN####
    # Index of the protein in the dataset
    fasta = full[['Target Sequence']].drop_duplicates().values.flatten() # fasta sequence on the dataset, in the same order as the dataset

    if fasta_protein not in fasta:
        print("The protein is not in the training dataset")
        # add this protein in fasta
        fasta = np.append(fasta,fasta_protein)

    try : 
        I_fasta = [int(dict_fasta2ind_all[fasta[i]]) for i in range(len(fasta))] # index of fasta in the precomputed dict and protein kernel, in the same order as the dataset
    except:
        print("The protein is not in the kernel")
        return None
    
    KP = KP_all[I_fasta,:][:,I_fasta]
    KP = torch.tensor(KP, dtype=mytype).to(device)
    print("kernel prot shape",KP.shape)

    # computation of feature for protein (no nystrom, just SVD)
    rP = KP.shape[0]#min(KP.shape[0],500)
    U, Lambda, VT = torch.svd(KP)
    Y = U[:,:rP] @ torch.diag(torch.sqrt(Lambda[:rP]))

    # nomramlisation of the features
    Y_c = Y - Y.mean(axis = 0)
    Y_cn = Y_c / torch.norm(Y_c,dim = 1)[:,None]
    print("protein features shape",Y.shape)

    # we test the protein on each drug
    df_test = pd.DataFrame()
    df_test['SMILES'] = list_smiles
    df_test['Target Sequence'] = fasta_protein
    df_test['Label'] = full.apply(lambda x:  x['Label'] if (x['SMILES'] in list_smiles) & (x['Target Sequence'] == fasta_protein) else -1 ,axis = 1)

    pred = np.zeros((len(list_smiles), len(train)))

    for i in range(len(train)):

        # compute the Nystrom approximation of the mol kernel and the features of the Kronecker kernel (features normalized and calculated on all mol contained in the dataset (train/val/test))
        X_cn = Nystrom_X_cn(mM,dM,nM,MorganFP)
        #print("mol features shape",X_cn.shape)

        #### TRAINING SET ####
        df_train = train[i].copy()

        # we add indices in df_test and df_train
        df_test['indfasta'] = df_test['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])
        df_train['indfasta'] = df_train['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])

        df_train['indsmiles'] = df_train['SMILES'].apply(lambda x: dict_smiles2ind[x])
        df_test['indsmiles'] = df_test['SMILES'].apply(lambda x: dict_smiles2ind[x])

        # Train
        I, J, y = load_datas(df_train)
        n = len(I)
        #print("len(train)",n)

        # Test
        I_test, J_test, y_test = load_datas(df_test)
        n_test = len(I_test)
        #print("len(test)",n_test)

        #### TRAINING ####
        # we train the model
        w_bfgs,b_bfgs,h = SVM_bfgs(X_cn,Y_cn,y,I,J,lamb,niter=50)
        # we compute a probability using weights (Platt scaling)
        s,t,h2 = compute_proba_Platt_Scalling(w_bfgs,X_cn,Y_cn,y,I,J,niter=20)
        #### TEST ####
        # we compute a probability using weights (Platt scaling)
        m,y_pred, proba_pred = compute_proba(w_bfgs,b_bfgs,s,t,X_cn,Y_cn,I_test,J_test)
        # we compute the results
        #acc1,au_Roc,au_PR,thred_optim,acc_best,cm,FP = results(y_test.cpu(),y_pred.cpu(),proba_pred.cpu())

        pred[:,i] = proba_pred.cpu().numpy()

    df_test['Proba_predicted_mean'] = np.mean(pred,axis = 1)
    df_test['Proba_predicted_std'] = np.std(pred,axis = 1)
    df_test['Proba_predicted_min'] = np.min(pred,axis = 1)
    df_test['Proba_predicted_max'] = np.max(pred,axis = 1)

    # sort the dataframe by the predicted probability
    df_test = df_test.sort_values(by = 'Proba_predicted_mean',ascending = False)
    #change Label on Label_known
    df_test = df_test.rename(columns = {'Label':'Label_known'})
    # remove indfasta and indsmiles, and Target sequence
    df_test = df_test.drop(columns = ['indfasta','indsmiles','Target Sequence'])

    return df_test