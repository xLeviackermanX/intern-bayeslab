import csv
import time
import json
import os
import random
import sys

# sys.path.append('/home/bayeslabs/New_Arch/')

from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from typing import Any, Callable, Dict, List, Optional, Tuple,Type

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, dataset, random_split, DataLoader
from torchmetrics import Accuracy

import pytorch_lightning as pl
from rdkit import Chem



import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask
from torch_geometric.data import DataLoader as GeoDataLoader



# from molflash.utils.preprocess import PreprocessingFunc,molgraph_collate_fn
from molflash.utils.emnPrepro import smile_to_graph, molgraph_collate_fn

from molflash.models.emn import EMN


class Dataset(Dataset):
    def __init__(self, list_ips, labels):
        self.list_ips = list_ips
        self.labels = labels

    def __getitem__(self, idx): 
        # input = PreprocessingFunc.fps_prep(self.list_ips[idx])
        input = self.list_ips[idx]
        
        target = self.labels[idx]
        sample = (input, target)
        return sample

    def __len__(self):
        print(len(self.list_ips))
        return len(self.list_ips)



class DataModule(pl.LightningDataModule):
    def __init__(self, filePath: str = None, transform = None, batch_size=None, splits=[0.8,0.1,0.1]):
        super().__init__()

        if filePath is None:
            raise ValueError("No Data Directory Provided.")
        self.transform = transform

        self.filePath = filePath
        self.batch_size = batch_size
        self.splits = splits



    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.filePath, nrows=1000)
        
        self.x = self.data['smiles']
        self.y = self.data['activity']


    def setup(self, stage: Optional[str] = None):
        # if stage == "fit" or stage in None:
        self.x = [self.transform(smi) for smi in self.x]
        self.dataset = [pair for pair in zip(self.x, self.y)]
            
        # self.train_data, self.val_data, self.test_data = self.dataset
            # print(self.train_data)
        train_len = int(self.splits[0]*len(self.dataset))
        test_len = int(self.splits[1]*len(self.dataset))
        val_len = len(self.dataset)-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_len, test_len, val_len])

        
        


        # if stage == "test" or stage is None: 
        #     self.test =

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:        
        return DataLoader(self.train_data, batch_size = self.batch_size,collate_fn=molgraph_collate_fn,num_workers=4)
        

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size = self.batch_size,collate_fn=molgraph_collate_fn,num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:        
        return DataLoader(self.test_data, batch_size = self.batch_size,collate_fn=molgraph_collate_fn,num_workers=4)

    """def predict_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.predict_data, batch_size = 5)"""


