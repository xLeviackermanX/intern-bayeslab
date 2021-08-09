import csv
import time
import json
import os
import random
import sys
from numpy.core.numeric import full

from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.functional import split
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split, sampler
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split


import rdkit
from rdkit import Chem

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.text.seq2seq.core.data import Seq2SeqFileDataSource

from  molflash.generator.utils.preprocess import LOG, create_vocabulary, Dataset, SMILESTokenizer, encode_property_change, get_parent_dir, save_df_property_encoded, split_data 
from  molflash.generator.utils.preprocess import get_parent_dir, split_data
import pickle


class GENDataModule(pl.LightningDataModule):
    def __init__(self, filePath: str = None, batch_size=16, collate_fn=None, split = None):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")

        self.filePath = filePath
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.split = split
        self.vocab = None
        self.tokenizer = SMILESTokenizer()

    def prepare_condition(self):
        self.property_change_encoder = encode_property_change(self.filePath)
        self.property_condition = []
        PROPERTIES = ['LogD', 'Solubility', 'Clint']
        for property_name in PROPERTIES:
            if property_name == 'LogD':
                intervals, _ = self.property_change_encoder[property_name]
                self.property_condition.extend(intervals)
            else:
                intervals = self.property_change_encoder[property_name]
                for name in intervals:
                    self.property_condition.append("{}_{}".format(property_name, name))
        self.encoded_file = save_df_property_encoded(self.filePath, self.property_change_encoder) 

    def prepare_data(self) -> None:
        self.prepare_condition()
        self.data = pd.read_csv(self.filePath)
        self.data = pd.unique(self.data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
  
        self.vocab = create_vocabulary(self.data, self.tokenizer, property_condition=self.property_condition)
        self.tokens = self.vocab.tokens()
 

        self.parent_path = get_parent_dir(self.filePath)
        output_file = os.path.join(self.parent_path, 'vocab.pkl')
        with open(output_file, 'wb') as pickled_file:
            pickle.dump(self.vocab, pickled_file)


        

    def setup(self, stage: Optional[str] = None):
        self.train, self.validation, self.test = split_data(self.encoded_file)

        # train_len = int(0.6*len(self.dataset))
        # test_len = int(0.2*len(self.dataset))
        # val_len = len(self.dataset)-train_len-test_len
        # self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_len, test_len, val_len])


        with open(os.path.join(self.parent_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pickle.load(input_file)

        self.train_data = Dataset(self.train, vocab, self.tokenizer, prediction_mode=False)
        self.val_data = Dataset(self.validation, vocab, self.tokenizer, prediction_mode=False)
        self.test_data = Dataset(self.test, vocab, self.tokenizer, prediction_mode=True)


    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size = self.batch_size,num_workers=4, collate_fn=self.collate_fn)





