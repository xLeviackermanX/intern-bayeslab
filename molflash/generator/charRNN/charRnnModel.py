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
from torchmetrics import Accuracy, F1
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.model import Task

from molflash.generator.charRNN.charRnnData import charRnnDataModule
from molflash.models.charRnn import CharRNN


class CharRnnTask(Task):
    """
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
    """

    def __init__(
        self,
        model: str = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
        
    ):

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate
        )
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x: Any) -> Any:
        '''
        input: prevs,lens,nexts
        --->Loss between nexts and charRnn outputs
        '''
        prevs, nexts, lens = x
        outputs, _, _ = self.model(prevs, lens)
        loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]),
                              nexts.view(-1))
        return outputs,loss

    
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs,loss = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        outputs, loss = self.forward(batch)
        self.log("valid_loss", loss)
        return loss


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        outputs, loss = self.forward(batch)
        self.log("test_loss", loss)
        return loss



if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)
    config = configFile.config

    dm = charRnnDataModule(config.filePath, batch_size = config.batch_size)
    dm.prepare_data()
    dm.setup()

    model = eval(config.modelName)(vocabulary=dm.vocab,config=config)

    model = CharRnnTask(model=model,loss_fn=eval(config.loss_fn))

    trainer = flash.Trainer(max_epochs=config.epochs,gpus=config.gpus, progress_bar_refresh_rate=20)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

