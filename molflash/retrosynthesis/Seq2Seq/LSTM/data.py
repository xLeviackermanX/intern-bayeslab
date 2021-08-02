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


class Prd2ReactDataModule(pl.LightningDataModule):
    """ A DataModule which helps in preparing data and creating dataloaders 

    Args:
    filePath : The filepath of the source data
    batch_size : Batch_size
    collate_fn : The collate function used.
        
    """
    def __init__(self, filePath: str = None, batch_size :int  = None, collate_fn = None, splits = None):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")

        self.filePath = filePath
        self.batch_size = batch_size
        self.collate_name = collate_fn
        if collate_fn is not None:
            self.collate_fn = eval(collate_fn)
        self.splits = splits

    def prepare_data(self) -> None:
        """ A Function to get the preprocessed data using filePath."""
        self.data, self.product_vocab_size, self.reactant_vocab_size, self.product_pad_index, self.reactant_pad_index = PreprocessingFunc.input_data(self.filePath)


    def setup(self, stage: Optional[str] = None):
        """ function to split data into train, test and valid."""
        
        fulldataset = len(self.data)
        train_len = int(self.splits[0]*fulldataset)
        test_len = int(self.splits[1]*fulldataset)
        val_len = fulldataset-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])

    # train DataLoader
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size = self.batch_size,num_workers=4)

    # val DataLoader
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,num_workers=4)

    # test DataLoader
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size = self.batch_size,num_workers=4)




