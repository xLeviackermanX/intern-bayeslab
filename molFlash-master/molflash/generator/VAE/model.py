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
import torch.nn.functional as F
from torchmetrics import Accuracy, F1
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers

from sklearn import datasets
from sklearn.model_selection import train_test_split

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask

from molflash.generator.VAE.data import VAEDataModule
from molflash.models.gru_encoder import VAEencoder
from molflash.models.gru_decoder import VAEdecoder

from molflash.utils.preprocess import get_vocabulary, tensor2string



class VAETask(flash.Task):
    """
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        model:  Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = nn.CrossEntropyLoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
        config = None,
        vocab = None,
    ):

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
           
        )
        self.optimizer=optimizer
        self.metrics=metrics
        self.learning_rate=learning_rate
        self.loss_fn = loss_fn
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.model = model
        self.config = config

        q_d_last = self.config.q_d_h * (2 if self.config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, self.config.d_z)
        self.q_logvar = nn.Linear(q_d_last, self.config.d_z)


    def encoder_forward(self, x):
        return self.encoder.forward(x)

    
    def decoder_forward(self, x, latent):
        return self.decoder(x, latent)
        

        
    def step(self, batch: Any, batch_idx: int) -> Any:

        z, kl_loss = self.encoder_forward(batch)
        recon_loss = self.decoder_forward(batch, z)

        loss = 0
        loss = kl_loss + recon_loss
        
       
        logs = {}
        logs["loss"] = loss
        outputs = {}
        outputs["loss"] = loss
        outputs["logs"] = logs

        return outputs

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)
        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)
        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            # z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.decoder.bos).repeat(n_batch)
            x = torch.tensor([self.decoder.pad]).repeat(n_batch,max_len)
            x[:, 0] = self.decoder.bos
            end_pads = torch.tensor([max_len]).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                    )

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.decoder.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder.decoder_rnn(x_input, h)
                y = self.decoder.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.decoder.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [tensor2string(i_x, self.vocab) for i_x in new_x]    


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_metric", outputs["metrics"])
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_metric", outputs["metrics"])

        return loss


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"test_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test_metric", outputs["metrics"])
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
    

    dm = VAEDataModule(filePath=config.filePath,batch_size=config.batch_size,splits=config.splits)
    dm.prepare_data()
    dm.setup()


    encoder_net = VAEencoder(dm.vocab, config)
    decoder_net = VAEdecoder(dm.vocab, config)


    model = VAETask(encoder = encoder_net, decoder = decoder_net, learning_rate=config.learning_rate, optimizer=eval(config.optimizer), config=config, vocab = dm.vocab)
    tb_logger = pl_loggers.TensorBoardLogger('logs/VAE')

    trainer = flash.Trainer(max_epochs=config.epochs,gpus=1, progress_bar_refresh_rate=20, logger=tb_logger)
    trainer.fit(model, datamodule=dm)
    trainer.test(model,datamodule=dm)


    # ckpt_file_path = '/home/bayeslabs/molFlash/molflash/generator/VAE/logs/VAE/default/version_0/checkpoints/epoch=12-step=4068.ckpt'
    # ckpt = torch.load(ckpt_file_path)

    # model.load_state_dict(ckpt["state_dict"])
    # print(model.sample(10))

