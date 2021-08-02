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


from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchmetrics

import rdkit 
from rdkit import Chem

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.data_module import DataModule
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys

from molflash.utils.preprocess import PreprocessingFunc

ND = np.ndarray


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        list_fps: Any,
        labels:int
    ):

        self.list_fps = list_fps

        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        fps = self.list_fps[index]
        label = self.labels[index]
        sample = (fps, label)

        return sample

    def __len__(self):
        return len(self.list_fps)
        

class PrdRxnDataSource(DataSource[Tuple[ND, ND]]):

    """ The DataSoure class which loads the data also loads predict data

    Args :
    data : The data to be loaded
    dataset : The dataset which can be used for loading data

    Result: The loaded data """

    def load_data(self, data: Any, dataset: ClassificationDataset):

        """ The function to load data for train, test and validation"""
        x, y = data
        ds = ClassificationDataset(x, y)
        return ds



class PrdRxnDataModule(flash.DataModule):
    
    """ The class to create datamodule for the task.

    Args : 
    path : The path of the data to be read
    preprocess : The preprocess class
    batch_size : The batch size
    num_workers : The number of CPU's to be used

    Results : Returns a DataModule object

    """
    @classmethod
    def get_data(cls, path: str = None) -> pd.DataFrame:

        """ function to read the data 

        Args : 
        path : Path of the data

        Results : Returns a DataFrame
        
        """
        return pd.read_csv(path, nrows=1000)

    @classmethod
    def from_dataset(cls, path, preprocess: Preprocess, batch_size: int = 4, num_workers: int = 0, valid_split: float = 0.2, test_size: float = 0.2):


        """ Class Function which gets data rom get_data and create datamodule"""
        cls.path = path
        data = cls.get_data(path)
        x1 = data['rxn_smiles']
        y1 = data['retro_templates']
        x, y = PreprocessingFunc.get_labels((x1, y1))
        preprocess = preprocess

        

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

        dm = cls.from_data_source(
            "numpy",
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=valid_split,
            )

        return dm






