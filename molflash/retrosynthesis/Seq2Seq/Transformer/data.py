import csv
import time
import json
import os
import random
import sys

from tqdm import tqdm
from numpy.core.numeric import full

from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type

import numpy as np
import pandas as pd

import torch
from torch.functional import split
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
    def __init__(self, filePath: str = None, batch_size=1, collate_fn=None, split = None):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")

        self.filePath = filePath
        self.batch_size = batch_size
        self.collate_name = collate_fn
        if collate_fn is not None:
            self.collate_fn = eval(collate_fn)
        self.split = split
        

    def prepare_data(self) -> None:
        self.data, self.product_vocab_size, self.reactant_vocab_size,  self.product_pad_index, self.reactant_pad_index = PreprocessingFunc.input_data(self.filePath)


    def setup(self, stage: Optional[str] = None):
        fulldataset = len(self.data)
        train_len = int(self.split[0]*fulldataset)
        test_len = int(self.split[1]*fulldataset)
        val_len = fulldataset-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, val_len, test_len])


    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size = self.batch_size,num_workers=4)
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size = self.batch_size,num_workers=4)


