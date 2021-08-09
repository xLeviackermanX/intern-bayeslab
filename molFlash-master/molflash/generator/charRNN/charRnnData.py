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
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split


from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Data

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys

from molflash.utils.preprocess import PreprocessingFunc
from molflash.generator.charRNN.utils.utils import CharVocab, string2tensor

ND = np.ndarray



class charRnnDataModule(pl.LightningDataModule):
    def __init__(self,filePath: str = None, batch_size: int = 16, splits: list = [0.7,0.15,0.15]):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")
        self.filePath = filePath
        self.batch_size = batch_size
        self.vocab=None
        self.splits = splits
    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.filePath)
        self.data = self.data['SMILES']
        self.vocab = self.get_vocabulary(self.data)
        torch.save(self.vocab, 'vocab.pt')
        

    def setup(self, stage: Optional[str] = None):
           
        train_len = int(self.splits[0]*len(self.data))
        test_len = int(self.splits[1]*len(self.data))
        val_len = len(self.data)-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])


    def collate(self,data):
        data.sort(key=len, reverse=True)
        tensors = [string2tensor(string,self.vocab)
                    for string in data]

        pad = self.vocab.pad
        prevs = pad_sequence([t[:-1] for t in tensors],
                                batch_first=True, padding_value=pad)
        nexts = pad_sequence([t[1:] for t in tensors],
                                batch_first=True, padding_value=pad)
        lens = torch.tensor([len(t) - 1 for t in tensors],
                            dtype=torch.long)
        return prevs, nexts, lens

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size = self.batch_size,collate_fn=self.collate,num_workers=4)
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,collate_fn=self.collate,num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size = self.batch_size,collate_fn=self.collate,num_workers=4)




