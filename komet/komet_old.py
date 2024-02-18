
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

from sklearn.metrics import  average_precision_score,  roc_curve, confusion_matrix, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
device_cpu = device
print(device)

mytype = torch.float16 # to save memory (only on GPU)
mytype = torch.float32

def load_df(name,dataset_dir="data"):
    """
    Load the data, clean smiles if RDkit cannot read them, and return the dataframe
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
    Add indices of the smiles to the dataframe
    """
    #### MOLECULE####
    # Index of the smiles in the dataset
    smiles = df[['SMILES']].drop_duplicates().values.flatten()
    nM = len(smiles)
    print("number of different smiles (mol):",nM)
    dict_smiles2ind = {smiles[i]:i for i in range(nM)}
    # add indsmiles in df
    df['indsmiles'] = df['SMILES'].apply(lambda x:dict_smiles2ind[x] )
    df = df.sort_values(by=['indsmiles'])
    df = df.reset_index(drop=True)
    return df, smiles

def Morgan_FP(list_smiles):
    """
    Compute the Morgan fingerprints of the molecules
    """
    ms = [Chem.MolFromSmiles(sm) for sm in list_smiles]
    nM = len(ms)
    MorganFP = np.zeros((nM,1024))
    for i in range(nM):
        # Generate Morgan fingerprint of the molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, nBits=1024)
        # Convert the fingerprint to a numpy array
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        MorganFP[i,:] = arr
    MorganFP = MorganFP.astype(int)
    MorganFP = torch.tensor(MorganFP, dtype=mytype).to(device)
    return MorganFP

def Nystrom_X(smiles_list,S,MorganFP,V,rM,Mu,epsi):
    """
    Compute the approximate features of the molecular kernel
    """
    # molecule kernel_first step : computation of Morgan fingerprint
    MorganFP_list= Morgan_FP(smiles_list)

    # compute the Nystrom approximation of the mol kernel and the features of the Kronecker kernel
    Z_list = ( MorganFP[S,:] @ MorganFP_list.T ) / ( 1024 - (1-MorganFP[S,:]) @ (1-MorganFP_list.T) )
    print("Z_list shape",Z_list.shape)

    X_list = Z_list.T @ V[:,:rM] @ torch.diag(1./torch.sqrt(epsi + Mu[:rM]))
    print("mol features list shape",X_list.shape)
    return X_list

def Nystrom_X_cn(k,rM,nM,MorganFP):
    """
    Compute the approximate kernel matrix of molecules
    """
    S = np.random.permutation(nM)[:k]
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

def add_indfasta(df):
    # Index of the protein in the dataset
    fasta = df[['Target Sequence']].drop_duplicates().values.flatten() # fasta sequence on the val dataset, in the same order as the dataset
    print("number of different Fasta (protein):",len(fasta))
    # add ind_fasta dans val, val et test
    df['indfasta'] = df['Target Sequence'].apply(lambda x: np.where(fasta==x)[0][0])
    return df, fasta  

def load_datas(df):
    """
    Load the data (indices of the pairs of molecules and proteins and the labels)
    """
    array_df = df[['indfasta','indsmiles','Label']].to_numpy()
    # get the indices of the interactions in the array_df
    J = array_df[:,0] # proteins
    I = array_df[:,1] # molecules
    y = array_df[:,2] # labels
    I = torch.tensor(I, dtype=torch.long).to(device)
    J = torch.tensor(J, dtype=torch.long).to(device)
    y = torch.tensor( np.sign(y-.5) ).to(device)
    return I, J, y


def SVM_bfgs(X_cn,Y_cn,y,I,J,lamb):
    """
    SVM model : train the SVM model 
    optimization of the Loss using a quasi-Newton method (L-BFGS)
    """
    n = len(I)
    XI = X_cn[I,:]
    def U(w): return torch.sum( (Y_cn@w)[J,:] * XI, axis=1 ) # FAST

    def Loss(u): return 1/n * torch.sum(torch.maximum(1+u,torch.tensor(0))) # loss function
    def g(w,b): return Loss(-y * (U(w)+b)) + lamb/2 * (w**2).sum() #function to monimize

    # L-BFGS
    def closure():
        lbfgs.zero_grad()
        objective = g(w_bfgs,b_bfgs)
        objective.backward()
        return objective

    rM = X_cn.shape[1]
    rP = Y_cn.shape[1]

    w_bfgs = torch.randn(rP,rM).to(device)
    b_bfgs = torch.randn(1).to(device)
    w_bfgs.requires_grad = True
    b_bfgs.requires_grad = True

    lbfgs = optim.LBFGS([w_bfgs,b_bfgs],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
    niter = 50
    history_lbfgs = []
    tic = time.perf_counter()
    for i in range(niter):
        history_lbfgs.append(g(w_bfgs,b_bfgs).item())
        lbfgs.step(closure)
    print(f"L-BFGS time: {time.perf_counter() - tic:0.4f} seconds")
    return w_bfgs, b_bfgs

def compute_proba_Platt_Scalling(w_bfgs,X_cn,Y_cn,y,I,J):
    """
    compute a probability using weights (Platt scaling)
    """
    n = len(I)
    XI = X_cn[I,:]
    def U(w): return torch.sum( (Y_cn@w)[J,:] * XI, axis=1 ) # FAST
    m = U(w_bfgs).detach() # do not forget to detach (do not backprop through it)!
    
    #logistic regression to find s and t
    def E(s,t): return 1/n * torch.sum( torch.log( 1+torch.exp(-y*(s*m+t)) ) )

    # L-BFGS
    def closure():
        lbfgs.zero_grad()
        objective = E(s,t)
        objective.backward()
        return objective
    s = torch.ones(1).to(device)
    t = torch.zeros(1).to(device)
    s.requires_grad = True
    t.requires_grad = True
    lbfgs = optim.LBFGS([s,t],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
    niter = 20
    history_lbfgs = []
    for i in range(niter):
        history_lbfgs.append(E(s,t).item())
        lbfgs.step(closure)

    return s,t

def compute_proba(w_bfgs,b_bfgs,s,t,X_cn,Y_cn,I,J):
    """
    compute a probability using weights
    """
    m = torch.sum( (Y_cn@w_bfgs)[J,:] * X_cn[I,:], axis=1 ) # do not forget to detach (do not backprop through it)!
    # comptute y_proba
    y_pred = torch.sign( m + b_bfgs ).detach()
    # compute the probability
    proba_pred = torch.sigmoid( (s*m + t) ).detach()
    return m,y_pred, proba_pred

def results(y,y_pred,proba_pred):
    """
    Compute the performance of the model (accuracy (threshold 0.5), AUC, average precision, optimal threshold, accuracy (best threshold), confusion matrix, false positive rate (best threshold))
    """

    # compute the accuracy (before Platt scalling, threshold 0.5)
    acc1 = torch.sum( y == y_pred ) / len(y)
    #print("accuracy (threshold 0.5)=",acc1.item())

    # compute the AUC
    fpr, tpr, thresholds = roc_curve(y,proba_pred)
    au_Roc = auc(fpr, tpr)
    #print("roc AUC:" + str(au_Roc))

    # compute the average precision
    au_PR = average_precision_score(y,proba_pred)
    #print("aupr=",au_PR)

    #optimal threshold
    precision = tpr / (tpr + fpr+0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    #print("optimal threshold: " + str(thred_optim))
    y_pred_s = [1 if i else -1 for i in (proba_pred >= thred_optim)]
    acc_best = torch.sum( y == torch.tensor(y_pred_s) ) / len(y)
    #print("accuracy (best threshold)=",acc_best.item())

    # confusion matrix
    cm = confusion_matrix(y, y_pred_s)
    #print('Confusion Matrix with best threshold: \n', cm)

    # False positive rate
    FP = cm[0,1]/(cm[0,1]+cm[0,0])
    #print('False Positive Rate with best threshold: ', FP)

    return acc1,au_Roc,au_PR,thred_optim,acc_best,cm,FP

