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
import csv
from logging import Logger

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

from molflash.retrosynthesis.Seq2Seq.LSTM.data import Prd2ReactDataModule
from molflash.models.Seq2Seq_retro import Seq2Seq,Encoder,Decoder

CUDA_LAUNCH_BLOCKING=1



class Prd2ReactTask(Task):
    """General Task for Sequence2Sequence.
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
        tgt_vocab_size : int = None,
    ):

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.tgt_vocab_size = tgt_vocab_size
        

    def forward(self, x: Any) -> Any:
        """ forward function for the encoder

        Args:
            x : The preprocessed input data 

        Returns: hidden and cell 
        """
        hidden,cell = self.encoder(x)
        
        return hidden,cell

    def dec_forward(self, x: Any, hidden: Any, cell: Any) -> Any:
        """ forward function for the decoder

        Args:
            x : The preprocessed input data 
            hidden : The hidden from the output of encoder
            cell : The cell from the output of encoder

        Returns:
            The probabilities of target sequence.
        """
        return self.decoder(x,hidden,cell)


    def step(self, batch: Any, batch_idx: int) -> Any:
        """ General step function 

        Args:
            batch : The input data in batches 
            batch_idx : The batch index

        Returns:
            The output which contains loss and metric results.
        """
        x, y = batch
        source = x
        target = y
        loss = torch.zeros(1, requires_grad=True).cuda()
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.tgt_vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, cell = self.forward(source)

        hidden = hidden[:,-batch_size:,:].contiguous()
        cell = cell[:,-batch_size:,:].contiguous()

  
        # Grab the first input to the Decoder which will be <SOS> token
        x_dec = target[:,0].unsqueeze(0)
        
        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.dec_forward(x_dec, hidden, cell)
            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            x_dec = best_guess.unsqueeze(0)

        y_hat = outputs

        metric = self.metrics(y_hat.argmax(-1).permute(1,0), y.cpu())

        y_hat = y_hat.reshape(x.shape[0],-1,x.shape[1])

        logs = {}
        output = {"y_hat": y_hat}
        output["y"] = y
        loss += self.loss_fn(y_hat,y.cpu())
        logs["loss"] = loss
        logs['metrics'] = metric
        output["metrics"] = metric

        output["logs"] = logs

        output["loss"] = loss
        return output


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_metric", outputs["metrics"])
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metric", outputs["metrics"])
        return loss


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
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

    # get congiguration
    configFile = OmegaConf.load(args.config)

    config = configFile.config


    dm = Prd2ReactDataModule(config.filePath, batch_size=config.batch_size, splits = config.splits)
    dm.prepare_data()
    dm.setup()

    
    encoder_net = Encoder(input_size = dm.product_vocab_size, embedding_size = config.enc_emb_size, hidden_size = config.enc_hid_size, num_layers = config.enc_n_layers
                    ,p = config.enc_drop)

    decoder_net = Decoder(input_size = dm.reactant_vocab_size, embedding_size = config.dec_emb_size, hidden_size = config.dec_hid_size, output_size = dm.reactant_vocab_size
                     ,num_layers = config.dec_n_layers, p = config.dec_drop)


    model = Prd2ReactTask(encoder=encoder_net, decoder=decoder_net,loss_fn = eval(config.loss_fn),optimizer=eval(config.optimizer),metrics = eval(config.metrics),tgt_vocab_size=dm.reactant_vocab_size)
    tb_logger = pl_loggers.TensorBoardLogger('logs/LSTM')

    trainer = flash.Trainer(max_epochs = config.epochs, gpus = config.gpus, logger=tb_logger)
    
    trainer.fit(model, datamodule=dm)


