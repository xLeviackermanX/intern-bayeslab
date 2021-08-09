#Basic imports
import csv
import time
import json
import os
import random
import sys
from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from rdkit import Chem

#torch imports
import torch
from pytorch_lightning import seed_everything
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, dataset, random_split, DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torch_geometric.data import DataLoader as GeoDataLoader

#custom imports
from molflash.utils.emnPrepro import smile_to_graph, molgraph_collate_fn
from molflash.models.emn import EMN


class emnDataset(Dataset):
    def __init__(self, list_ips, labels):
        # Load the Dataset
        self.list_ips = list_ips
        self.labels = labels

    def __getitem__(self, idx):
        # get sample by a given index
        input = self.list_ips[idx]

        target = self.labels[idx]
        sample = (input, target)
        return sample

    def __len__(self):
        # get length of the dataset
        print(len(self.list_ips))
        return len(self.list_ips)


class emnDataModule(pl.LightningDataModule):
    """
    EMN DataModule
    """
    def __init__(self, filePath: str = None, transform=None, batch_size=None, splits=[0.8, 0.1, 0.1]):
        """

        :param filePath: input classification data file path
        :param transform: EMN preprocessing functions
        :param batch_size: batch of training for training
        :param splits: splits of data ranges
        """
        super().__init__()
        if filePath is None:
            raise ValueError("No Data Directory Provided.")
        self.transform = transform
        self.filePath = filePath
        self.batch_size = batch_size
        self.splits = splits

    def prepare_data(self) -> None:
        """
        how to load the data and tokenize
        """
        self.data = pd.read_csv(self.filePath)
        self.x = self.data['smiles']
        self.y = self.data['activity']

    def setup(self, stage: Optional[str] = None):
        """
        how to split the data and preprare train, val, test dataset
        """
        self.x = [self.transform(smi) for smi in self.x]
        self.dataset = [pair for pair in zip(self.x, self.y)]
        train_len = int(self.splits[0] * len(self.dataset))
        test_len = int(self.splits[1] * len(self.dataset))
        val_len = len(self.dataset) - train_len - test_len
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_len, test_len, val_len])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=molgraph_collate_fn, num_workers=4)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=molgraph_collate_fn, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=molgraph_collate_fn, num_workers=4)

