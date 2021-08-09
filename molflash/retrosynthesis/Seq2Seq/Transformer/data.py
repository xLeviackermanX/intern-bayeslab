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
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys

from flash.text.seq2seq.core.data import Seq2SeqFileDataSource

from molflash.utils.preprocess import PreprocessingFunc
from molflash.retrosynthesis.utils.Seq2Seq_preprocess import data_process, build_vocab, tokenize

class Prd2ReactDataModule(pl.LightningDataModule):
    """ A DataModule which helps in preparing data and creating dataloaders 

    Args:
    filePath : The filepath of the source data
    batch_size : Batch_size
    collate_fn : The collate function used.
        
    """
    def __init__(self, filePath: str = None, batch_size :int  = None, splits = [0.8,0.1,0.1]):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")

        self.filePath = filePath
        self.batch_size = batch_size
        self.source_vocab = None
        self.target_vocab = None
        self.splits = splits
        self.tokenizer = tokenize

    def prepare_data(self) -> None:
        """ A Function to get the preprocessed data using filePath."""
        self.source_vocab = build_vocab(self.filePath[0], tokenizer=self.tokenizer)
        self.target_vocab = build_vocab(self.filePath[1], tokenizer=self.tokenizer)
        self.data = data_process(self.filePath, self.source_vocab, self.target_vocab, self.tokenizer)


    def setup(self, stage: Optional[str] = None):
        """ function to split data into train, test and valid."""
        
        fulldataset = len(self.data)
        train_len = int(self.splits[0]*fulldataset)
        test_len = int(self.splits[1]*fulldataset)
        val_len = fulldataset-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])

    # train DataLoader
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)

    # val DataLoader
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)

    # test DataLoader
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)

    def collate_fn(self, data_batch):
        prod_batch=[torch.cat([torch.tensor([self.source_vocab['<bos>']]), prod_item, torch.tensor([self.source_vocab['<eos>']])], dim=0) for (prod_item, react_item) in data_batch]
        react_batch=[torch.cat([torch.tensor([self.target_vocab['<bos>']]), react_item, torch.tensor([self.target_vocab['<eos>']])], dim=0) for (prod_item, react_item) in data_batch]
        prod_batch = pad_sequence(prod_batch, padding_value=self.source_vocab['<pad>'])
        react_batch = pad_sequence(react_batch, padding_value=self.target_vocab['<pad>'])
        return prod_batch, react_batch


