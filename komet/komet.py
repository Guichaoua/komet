import pandas as pd
import numpy as np
import pickle
import zipfile

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.optim as optim
from sklearn import svm
import time

import psutil

from sklearn.metrics import average_precision_score, roc_curve, confusion_matrix, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
device_cpu = device
print(device)

mytype = torch.float16  # to save memory (only on GPU)
mytype = torch.float32

def load_df(name):
    """
    Loads a dataframe from a CSV file, cleans up SMILES strings that cannot be read by RDKit,
    and returns the cleaned dataframe.

    :param name: The name of the file (with extension) to be loaded. If the file is a zip archive,
                 it will be extracted first.
    :type name: str
    :return: The cleaned dataframe with SMILES strings that RDKit can read.
    :rtype: pd.DataFrame
    """
    # If the data is in a zip file, unzip it
    if ".zip" in name:
        with zipfile.ZipFile(f"data/{name}", 'r') as zip_ref:
            zip_ref.extractall("data/")
        name = name[:-4]
    df = pd.read_csv(f"data/{name}", index_col=0)
    # Clean smiles
    smiles = df[['SMILES']].drop_duplicates().values.flatten()
    l_smiles = [sm for sm in smiles if Chem.MolFromSmiles(sm) is None]
    print(f"number of smiles to clean: {len(l_smiles)}")
    df = df[~df['SMILES'].isin(l_smiles)]
    print(f"{name} shape", df.shape)
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
    dict_ind2smiles = {i: smiles[i] for i in range(nM)}
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

def SVM_bfgs(X_cn, Y_cn, y, I, J, lamb):
    """
    Trains an SVM model using the L-BFGS optimization algorithm to minimize the loss function.

    :param X_cn: Feature matrix for compounds.
    :param Y_cn: Feature matrix for targets.
    :param y: Interaction labels.
    :param I: Indices of molecules.
    :param J: Indices of proteins.
    :param lamb: Regularization parameter.
    :return: Optimized weight and bias parameters for the SVM model.
    :rtype: tuple[torch.Tensor, torch.Tensor]
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
    niter = 50
    history_lbfgs = []
    tic = time.perf_counter()
    for i in range(niter):
        history_lbfgs.append(g(w_bfgs, b_bfgs).item())
        lbfgs.step(closure)
    print(f"L-BFGS time: {time.perf_counter() - tic:0.4f} seconds")
    return w_bfgs, b_bfgs

def compute_proba_Platt_Scalling(w_bfgs, X_cn, Y_cn, y, I, J):
    """
    Computes probability estimates for interaction predictions using Platt scaling.

    :param w_bfgs: Optimized weights from the SVM model.
    :param X_cn: Feature matrix for compounds.
    :param Y_cn: Feature matrix for targets.
    :param y: Interaction labels.
    :param I: Indices of molecules.
    :param J: Indices of proteins.
    :return: Optimized parameters 's' and 't' for Platt scaling.
    :rtype: tuple[torch.Tensor, torch.Tensor]
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
    niter = 20
    history_lbfgs = []
    for i in range(niter):
        history_lbfgs.append(E(s, t).item())
        lbfgs.step(closure)

    return s, t


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
