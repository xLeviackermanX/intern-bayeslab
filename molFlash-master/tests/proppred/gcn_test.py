import csv
import time
import json
import os
import random
import sys

import pytorch_lightning
from rdkit import Chem

import torchmetrics

from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, F1


import flash
from flash.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from flash.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask

from molflash.proppred.classification.data import GCNDataModule, GCNPreprocess
from molflash.utils.preprocess import PreprocessingFunc


from molflash.models.gcn import GCN
from pytorch_lightning.loggers import TensorBoardLogger
from molflash.proppred.classification.model import ClassificationTask



if __name__ == "__main__":
    """
    sample training
    """
    # x, y = dd["smiles"], dd["activity"]
    # ab = dd.train_dataloader()

    logger = TensorBoardLogger("tb_logs", name="model")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)

    preprocessing = configFile.config.preprocessing
    filePath = configFile.config.filePath
    dropout = configFile.config.dropout
    optimizer = configFile.config.optimizer
    loss_fn = configFile.config.loss_fn
    batch_size = configFile.config.batch_size
    epochs = configFile.config.epochs
    split = configFile.config.split
    gpus = configFile.config.gpus

    dm = GCNDataModule.from_dataset(filePath, GCNPreprocess())
    model = ClassificationTask(GCN(in_channel=40, hid1=128, hid2=256, hid3=64, lin1=128, lin2=128, out=1, drop=0.5),
                               optimizer=optim.Adam, learning_rate=10e-3,
                               metrics=torchmetrics.Accuracy())

    trainer = flash.Trainer(max_epochs=5, progress_bar_refresh_rate=20,gpus=0)
    trainer.fit(model, datamodule=dm)
