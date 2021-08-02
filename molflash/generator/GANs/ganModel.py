import csv
import time
import json
import os
import random
import sys
import pytorch_lightning

from pytorch_lightning.core.datamodule import LightningDataModule

sys.path.append('/home/bayeslabs/molFlash/')


from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, F1
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler



import flash
from flash.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from flash.core.model import Task
from flash.text.seq2seq.translation.metric import BLEUScore

from molflash.generator.GANs.ganData import GANDataModule
from molflash.models.latentGAN import Sampler, Generator, Discriminator, LatentGAN, compute_gradient_penalty
import torch.autograd as autograd
from torch.autograd import Variable

CUDA_LAUNCH_BLOCKING=1



class GANTask(Task):
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
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5
    ):

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,

        )
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
    


    def step(self, batch: Any, batch_idx: int) -> Any:
        """ General step function 

        Args:
            batch : The input data in batches 
            batch_idx : The batch index

        Returns:
            The output which contains loss and metric results.
        """

        n_batch = 10

        g_opt, d_opt = self.optimizers()

        X = batch
        batch_size = X.shape[0] 

        # batch_size = torch.from_numpy(batch_size)

        # print(type(batch_size))
        # exit()

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.generator(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.discriminator(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.discriminator(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = (errD_real + errD_fake)

        d_opt.zero_grad()
        self.backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.discriminator(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.backward(errG)
        g_opt.step()
        outputs = {}
        logs = {}
        logs['g_loss'] = errG
        logs['d_loss'] = errD
        outputs['logs'] = logs
        print(logs)
        exit()

        self.log_dict({'g_loss': errG, 'd_loss': errD}, prog_bar=True)
        
        return outputs
        # self.S = Sampler(self.generator)    
        # fake_mols = self.S.sample(n_batch)


        # real_validity = self.discriminator(batch)
        # fake_validity = self.discriminator(fake_mols)

        # gradient_penalty = compute_gradient_penalty(batch, fake_mols, self.discriminator)

        # d_loss = -torch.mean(real_validity) \
        #         + torch.mean(fake_validity) \
        #         + 10 * gradient_penalty

        # loss = Variable(d_loss, requires_grad = True)

        # # loss = 0
        # print(d_loss)
        # exit()
        # # loss.append(d_loss.item())
        # print(loss)
        # logs = {}
        # output = {}
        # # loss += self.loss_fn(y_hat,y)
        


        # logs["loss"] = loss
        # output["logs"] = logs

        # output["loss"] = loss

        # return output


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        # real_img = batch
        # batch_size = real_img.shape[0]
        # z = torch.randn(batch_size, self.latent_dim).to(self.device)

        # # Train Discriminator 
        # if optimizer_idx == 0:
        #     fake_img = self.generator(z)
        #     fake_pred = self.discriminator(fake_img)
        #     real_pred = self.discriminator(real_img)
        #     d_loss = losses.d_loss(real_pred, fake_pred)
        #     if batch_idx % self.args.d_regularize_every == 0:
        #         real_img.requires_grad = True
        #         real_pred = self.discriminator(real_img)
        #         self.d_reg_loss = losses.d_reg_loss(real_pred, real_img)
        #     return {'loss': d_loss}

        # # Train Generator
        # if optimizer_idx == 1:
        #     fake_img = self.generator(z)
        #     fake_pred = self.discriminator(fake_img)
        #     g_loss = losses.g_loss(fake_pred)
        #     if batch_idx % self.args.g_regularize_every == 0:
        #         self.g_reg_loss = losses.g_reg_loss(fake_img, z)
        #     return {'loss':g_loss}
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"test_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5)
        return g_opt, d_opt

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
    #  # Step using d_loss or g_loss
    # #  super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
    #  if optimizer_idx == 0:
    #      self.discriminator.zero_grad()
    #      self.d_reg_loss.backward()
    #     #  super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
    #  if optimizer_idx == 1:
    #      self.generator.zero_grad()
    #      self.g_reg_loss.backward()
    #     #  super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)


if __name__=="__main__":


    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)

    preprocessing = configFile.config.preprocessing
    filePath = configFile.config.filePath
    latent_size = configFile.config.latent_dim     
    # hidden_size = configFile.config.hid_size  
    # num_layers = configFile.config.n_layers   
    # n_heads = configFile.config.n_heads       
    dropout = configFile.config.dropout
    optimizer = configFile.config.optimizer
    loss_fn = configFile.config.loss_fn
    batch_size = configFile.config.batch_size
    epochs = configFile.config.epochs
    split = configFile.config.splits
    gpus = configFile.config.gpus


    dm = GANDataModule(filePath, batch_size=batch_size, split=split)
    dm.prepare_data()
    dm.setup()

    print(dm)

    gen = Generator()
    dis = Discriminator()

    model = GANTask(gen, dis,loss_fn = eval(loss_fn))
    tb_logger = pl_loggers.TensorBoardLogger('logs/')


    trainer = flash.Trainer(max_epochs=epochs,gpus=gpus, logger=tb_logger)
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)






