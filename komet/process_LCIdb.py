import argparse
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as D



def process_LCIdb(name_file, data_dir = "./preprocessed/", max_length_fasta = 1000, bioactivity_choice = "checkand1database",min_weight = 100, max_weight = 900,  interaction_plus = 1e-7, interaction_minus = 1e-4):
    """
    Processes data from a given ligand-chemical interaction database file and performs various data cleaning and transformation steps.

    :param name_file: The name of the file containing the database information.
    :type name_file: str
    :param data_dir: The directory where the database file and the file prot_uniprot_fasta_all.csv are located, defaults to "./preprocessed/".
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

    def calcul_score_minMax(i,x):
        m = "min "+x
        M = "max "+x
        if df.loc[i,m] >= score_plus:
            return 1
        elif df.loc[i,M] <= score_minus:
            return 0
        else:
            return 0.5

    def f_score(i):
        # If a given compound and protein target have multiple measurements of different types,
        # we choose them in the following order of preference: Kd over Ki over IC50.
        if not np.isnan(df.loc[i,"mean pKd"]):
            return calcul_score_mean(df.loc[i,"mean pKd"])
        elif not np.isnan(df.loc[i,"mean pKi"]):
            return calcul_score_mean(df.loc[i,"mean pKi"])
        elif not np.isnan(df.loc[i,"mean pIC50"]):
            return calcul_score_mean(df.loc[i,"mean pIC50"])
        else:
            if not np.isnan(df.loc[i,"min pKd"]):
                return calcul_score_minMax(i,"pKd")
            elif not np.isnan(df.loc[i,"min pKi"]):
                return calcul_score_minMax(i,"pKi")
            elif not np.isnan(df.loc[i,"min pIC50"]):
                return calcul_score_minMax(i,"pIC50")
            else:
                return np.nan

    df['score'] = list(map(lambda i: f_score(i), df.index))
    print("score calculé")
    print("df.shape = ",df.shape)

    # register the dataframe
    df.to_csv(data_dir+"LCIdb.csv", index = False)

    return df





if __name__ == "__main__":

    parser = argparse.ArgumentParser("This script processes the BindingDB \
    database in order to have the drugs, the proteins and their \
    interactions with these filters:\n\
    - small proteins\n\
    - molecules with know Smiles, µM between 100 and 900\n\
    - proteins with all known aa in list, known fasta, and length < 1000\n")

    
    
    parser.add_argument("name_file", type = str,
        help = "Name of the file .csv download from https://zenodo.org/record/6398019#.Y6A4nrKZPn4 , for exemple'Consensus_CompoundBioactivity_Dataset_v1.1.csv'")

    parser.add_argument("--data_dir", type = str,
        help = "Name of directory where the files .csv is, for exemple 'data/'")
    
    parser.add_argument("--max_length_fasta", type = int,
        help = "max length of the fasta for proteins, for exemple 1000")
    
    parser.add_argument( "--bioactivity_choice", type = str,
        help = "name of the choice for bioactivity, for exemple 'check' or 'check_and_1database' or 'all'",
        choices=['check', 'check_and_1database', 'all'])
    
    parser.add_argument("--min_weight", type = int,
        help = "min weight of the molecule, for exemple 100")
    
    parser.add_argument("--max_weight", type = int,
        help = "max weight of the molecule, for exemple 900")
    
    parser.add_argument("--interaction_plus", type = float,
        help = "interactions + (1) if measure (Kd,Ki,IC50) < 1e-7M, for exemple 1e-7")
    
    parser.add_argument("--interaction_minus", type = float,
        help = "interaction - (0), if measure (Kd,Ki,IC50) > 1e-4M for exemple 1e-4")

    args = parser.parse_args()



    # Process of the CC database (molecules with know Smiles, µM between 100 and 900\n\
    # and proteins with all known aa in list, known fasta, and length < 1000\n")
    
    process_LCIdb(args.name_file, args.data_dir, args.max_length_fasta, args.bioactivity_choice, args.min_weight, args.max_weight,  args.interaction_plus, args.interaction_minus)
    print("process LCIdb done")

