#Basic imports
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

#torch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Dataset, random_split

# chem packages
import rdkit
from rdkit import Chem

#custom imports
from molflash.utils.preprocess import PreprocessingFunc
from molflash.models.gcn import GCN

CUDA_LAUNCH_BLOCKING=1

class gcnDataset(Dataset):
    def __init__(self, data, transform=None):
        # Load the Dataset
        self.x,self.y = zip(*data)#data['smiles']
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


class gcnPreprocess:
    """ All Preprocessing and
        Cleaning of Data sample is done """

    def __init__(self):
        super().__init__()

    def get_graphfeatures(sample: Any) -> Any:
        # convert the given smile to graph features [x, edge_index, label]
        smile, y = sample
        mol = Chem.MolFromSmiles(smile)
        edge_index, x = PreprocessingFunc.graph_representation(mol, 40)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1, -1)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data


class gcnDataModule(pl.LightningDataModule):
    def __init__(self, splits, filepath: str = None, batch_size: int = 10, num_workers: int = 4):
        """
        initialize transforms and input filepath
        :param preprocess: preprocess function
        :param splits: split ranges
        :param filepath: input filepath
        :param batch_size: size of each input batch
        :param num_workers: dataloader worker
        """

        super().__init__()
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filepath = filepath

    def prepare_data(self):
        """
        how to load the data and tokenize
        """

        self.data = pd.read_csv(self.filepath)
        return self.data

    def clean_data(self,threshold: int = 40):
        """
        Remove the invalid smiles and  get only smiles having atoms threshold less than 40
        :param threshold:
        :return:  dataset
        """
        data=self.data
        for index, row in data.iterrows():
            try:
                molecule = Chem.MolFromSmiles(data["smiles"][index])
                n_atoms = molecule.GetNumAtoms()
            except:
                data.drop(index, inplace=True)
        data["num_atoms"] = [Chem.MolFromSmiles(smile).GetNumAtoms() for smile in data['smiles']]
        data = data[data["num_atoms"] <= threshold]
        data = data.drop("num_atoms", axis=1)
        data = data.reset_index()
        return data

    def setup(self):
        """
        how to split the data and preprare train, val, test dataset
        """
        self.dataset = pd.read_csv(self.filepath)
        self.dataset = self.clean_data()
        length=len(self.dataset)
        smiles = list(self.dataset['smiles'])
        activity = list(self.dataset['activity'])
        self.dataset = list(zip(smiles,activity))

        train_len = int(self.splits[0] * length)
        test_len = int(self.splits[1] * length)
        val_len = length - train_len - test_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_len, test_len, val_len])

        self.train_dataset = gcnDataset(self.train_dataset, transform=PreprocessingFunc.get_graphfeatures1)
        self.val_dataset = gcnDataset(self.val_dataset, transform=PreprocessingFunc.get_graphfeatures1)
        self.test_dataset = gcnDataset(self.test_dataset, transform=PreprocessingFunc.get_graphfeatures1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)