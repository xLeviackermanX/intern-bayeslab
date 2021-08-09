import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from moltorch.models import MPNN
from torch.utils.data import Dataset, DataLoader




class PreprocessingFunc:

    def __init__(self):
        self.amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
               'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
        self.enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

        self.MAX_SEQ_PROTEIN = 1000

    def trans_protein(self,x):
        temp = list(x.upper())
        temp = [i if i in amino_char else '?' for i in temp]
        if len(temp) < self.MAX_SEQ_PROTEIN:
            temp = temp + ['?'] * (self.MAX_SEQ_PROTEIN-len(temp))
        else:
            temp = temp [:self.MAX_SEQ_PROTEIN]
        return temp

    def protein_2_embed(self,x):
        return self.enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

    def preprocess_data(path, frac=[0.7, 0.3, 0.0]):
        # '/home/praveen/pk/virtualscreening/trying/df.tsv'
        data = pd.read_csv(path, delimiter='\t')
        print('Length of the Data', len(data))
        data_list = []

        unique = pd.Series(data['SMILES'].unique()).apply(smiles2mpnnfeature)
        unique_dict = dict(zip(data['SMILES'].unique(), unique))
        data['drug_encoding'] = [unique_dict[i] for i in data['SMILES']]

        AA = pd.Series(data['Target Sequence'].unique()).apply(trans_protein)
        AA_dict = dict(zip(data['Target Sequence'].unique(), AA))
        data['target_encoding'] = [AA_dict[i] for i in data['Target Sequence']]

        df = data[['drug_encoding', 'target_encoding', 'Label']]
        train = df.sample(n=int(len(data) * frac[0]))
        df.drop(train.index.valueclass PreprocessingFunc:
