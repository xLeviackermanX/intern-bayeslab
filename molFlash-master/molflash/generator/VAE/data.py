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

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, dataset, random_split, DataLoader
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask

from molflash.utils.preprocess import get_vocabulary, string2tensor



class Dataset(Dataset):
    def __init__(self, list_ips, labels):
        self.list_ips = list_ips
        self.labels = labels

    def __getitem__(self, idx): 
        input = self.list_ips[idx]
        target = self.labels[idx]
        sample = (input, target)
        return sample

    def __len__(self):
        return len(self.list_ips)



class VAEDataModule(pl.LightningDataModule):
    def __init__(self, filePath: str = None, vocab = None, batch_size=None, splits=[0.8,0.1,0.1]):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")

        self.filePath = filePath
        self.batch_size = batch_size
        self.splits = splits
        self.vocab = None



    def prepare_data(self) -> None:
        self.data = list(pd.read_csv(self.filePath, nrows=50000)['SMILES'])
        self.vocab = get_vocabulary(self.data)


    def setup(self, stage: Optional[str] = None):


        train_len = int(self.splits[0]*len(self.data))
        test_len = int(self.splits[1]*len(self.data))
        val_len = len(self.data)-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])



    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:        
        return DataLoader(self.train_data, batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)
        

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:        
        return DataLoader(self.test_data, batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    
    def collate_fn(self, data):
        data.sort(key=len, reverse = True)
        tensors = [string2tensor(string, self.vocab)for string in data]
        return tensors


