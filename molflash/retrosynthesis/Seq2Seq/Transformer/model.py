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
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

from sklearn import datasets
from sklearn.model_selection import train_test_split

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.model import Task
from flash.text.seq2seq.translation.metric import BLEUScore

from molflash.retrosynthesis.Seq2Seq.Transformer.data import Prd2ReactDataModule
from molflash.models.Seq2Seq_transformer import Translation
from molflash.models.transformer import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask

CUDA_LAUNCH_BLOCKING=1



class Prd2ReactTask(Task):
    """General Task for Sequence2Sequence.
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        val_target_max_length: Maximum length of targets in validation. Defaults to `128`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.01,
        src_pad_index = None,
        tgt_pad_index = None,
        tgt_vocab_size = None,
    ):


        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,


        )
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.tgt_vocab_size = tgt_vocab_size


    def step(self, batch: Any, batch_idx: int) -> Any:
        """ General step function 

        Args:
            batch : The input data in batches 
            batch_idx : The batch index

        Returns:
            The output which contains loss and metric results.
        """

        x, y = batch


        batch_size = x.shape[0]
        seq_len = x.shape[1]

        target_len = seq_len

        outputs = torch.zeros(target_len, batch_size, self.tgt_vocab_size).cuda()


        tgt_input = y[:,0].unsqueeze(1)
        
        for t in range(0, target_len):

            output = self.model(x,tgt_input)

            output = output.permute(1,0,2)

            outputs[t] = output

            best_guess = output.argmax(-1)

            best_guess = best_guess.permute(1,0)

            tgt_input = best_guess

        y_hat = outputs

        loss = 0
        metric = self.metrics(y_hat.permute(1,0,2).argmax(-1),y.int())

        logs = {}

        output = {"y_hat": y_hat}
        output["y"] = y
        loss += self.loss_fn(y_hat.permute(1,2,0),y)
        logs["loss"] = loss
        output["metrics"] = metric
        logs['metrics'] = metric
        
        output["loss"] = loss

        output["logs"] = logs


        return output


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_metric", outputs["metrics"])
        return loss


    def common_step(self, prefix: str, batch: Any) -> Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)


    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metric", outputs["metrics"])
        return loss


    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"test_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_metric", outputs["metrics"])
        return loss


    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = self.optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer_kwargs["lr"] = self.learning_rate
            optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_kwargs)
        if self.scheduler:
            return [optimizer], [self._instantiate_scheduler(optimizer)]
        return optimizer



if __name__=="__main__":


    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    configFile = OmegaConf.load(args.config)
    config = configFile.config

    dm = Prd2ReactDataModule(config.filePath, batch_size=config.batch_size, split=config.split)
    dm.prepare_data()
    dm.setup()


    transformer_model = Translation(src_vocab_size=dm.product_vocab_size,tgt_vocab_size=dm.reactant_vocab_size,src_pad_index = dm.product_pad_index,
                 tgt_pad_index = dm.reactant_pad_index)

    src_pad_index=dm.product_pad_index
    
    model = Prd2ReactTask(model = transformer_model,loss_fn = eval(config.loss_fn),optimizer=eval(config.optimizer), metrics = eval(config.metrics), src_pad_index = dm.product_pad_index, tgt_pad_index = dm.reactant_pad_index, tgt_vocab_size = dm.reactant_vocab_size) 
    tb_logger = pl_loggers.TensorBoardLogger('logs/Transformer')


    trainer = flash.Trainer(max_epochs=config.epochs,gpus=1, logger=tb_logger)
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)
    
    
    


