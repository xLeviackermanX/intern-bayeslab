import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Dataset,random_split

import rdkit
from rdkit import Chem

from molflash.utils.preprocess import PreprocessingFunc


class CPPDataset(Dataset):
    def __init__(self, data, transform=None):
        # Load the whole Dataset
        self.x, self.y = zip(*data)
        self.num_samples = len(self.y)
        self.transform = transform

    def __getitem__(self, index: int = 0):
        # get sample by a given index

        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # get length of the dataset
        return self.num_samples


class CPPPreprocess:
    """ All Preprocessing and
        Cleaning of Data sample is done """

    def __init__(self):
        super().__init__()

    def get_graphfeatures(sample: Any) -> Any:
        # convert the given smile to graph features [x, edge_index, label]
        tuplesmiles, y = sample
        smile1 = tuplesmiles[0]
        smile2 = tuplesmiles[1]
       
        mol = Chem.MolFromSmiles(smile1)
        mol1 = Chem.MolFromSmiles(smile2)
        edge_index, x = PreprocessingFunc.graph_representation(mol, 40)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        
        edge_index1, x1 = PreprocessingFunc.graph_representation(mol1, 40)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long)
        x1 = torch.tensor(x1, dtype=torch.float)
        
        y = torch.tensor(y, dtype=torch.float).view(1, -1)
        data = Data(x=x, edge_index=edge_index,x1=x1, edge_index1=edge_index1, y=y)
        return data


class CPPDataModule(pl.LightningDataModule):
    def __init__(self, preprocess, splits, filepath: str = None, batch_size: int = 10, num_workers: int = 1):
        # initialize transforms and input filepath
        super().__init__()
        self.preprocess = preprocess
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filepath = filepath

    def prepare_data(self):
        # how to load the data and tokenize
        self.data = pd.read_csv(self.filepath, nrows=100)
        return self.data

    def clean_data(self,threshold: int = 40):
        # Remove the invalid smiles and  get only smiles having atoms threshold less than 40
        data=self.data
        X,Y = PreprocessingFunc.getFwdLabels(self.data)
        data_filtered = []
        for idx in range(len(X)):
            p,r = X[idx]
            if Chem.MolFromSmiles(p).GetNumAtoms() <= threshold and Chem.MolFromSmiles(r).GetNumAtoms() <= threshold:
                data_filtered.append([X[idx],Y[idx]])
        return data_filtered

    def setup(self):
        # how to split the data
        self.dataset = self.prepare_data()
        self.dataset = self.clean_data()
        
        train_len = int(self.splits[0]*len(self.dataset))
        test_len = int(self.splits[1]*len(self.dataset))
        val_len = len(self.dataset)-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_len, test_len, val_len])
        
        

        self.train_dataset = CPPDataset(self.train_data, transform=eval("PreprocessingFunc." + str(self.preprocess)))
        self.val_dataset = CPPDataset(self.val_data, transform=eval("PreprocessingFunc." + str(self.preprocess)))
        self.test_dataset = CPPDataset(self.test_data, transform=eval("PreprocessingFunc." + str(self.preprocess)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=30)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=30)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=30)

