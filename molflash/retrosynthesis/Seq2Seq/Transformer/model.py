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
from torch._C import device

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

from molflash.utils.preprocess import PreprocessingFunc
from molflash.retrosynthesis.Seq2Seq.Transformer.new_data import Prd2ReactDataModule

from molflash.models.transformer import Seq2SeqModel, Encoder, Decoder
from molflash.retrosynthesis.utils.Seq2Seq_preprocess import predict_preprocess, tokenize

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
        vocab = None,
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
        self.vocab = vocab


    def step(self, batch: Any, batch_idx: int) -> Any:
        """ General step function 

        Args:
            batch : The input data in batches 
            batch_idx : The batch index

        Returns:
            The output which contains loss and metric results.
        """

        x, y = batch

        x = x.permute(1,0)
        y = y.permute(1,0)


        decoder_output = self.model(x,y[:,:-1])
    
        target = y[:,1:]

        y_hat = decoder_output

        loss = 0
        metric = self.metrics(y_hat.argmax(-1),target.int())    #batch,Seq_len,vocab

        logs = {}

        output = {"y_hat": y_hat}
        output["y"] = target

        loss += self.loss_fn(y_hat.permute(0,2,1), target)    #batch,vocab,seq_len

        logs["loss"] = loss
        output["metrics"] = metric
        logs['metrics'] = metric
        
        output["loss"] = loss

        output["logs"] = logs


        return output

    def predict(self, x : Any, max_length = 100):

        with torch.no_grad():

            source = predict_preprocess(x, self.vocab, tokenize)
            print(source)

            source = source.permute(1,0)
            # source = torch.LongTensor(source).unsqueeze(0)

            source_mask = self.model.get_source_mask(source)

            encoder_output = self.model.encoder_stack(source,source_mask)

            target_bos = [self.vocab['<bos>']]
            target_eos = self.vocab['<eos>']
            # target_to_idx = []

            i = 0
            while i < max_length:
        
                #step 8.1 convert the current output sentence prediction into a tensor with a batch dimension
                # target_bos = torch.LongTensor(target_bos).unsqueeze(0)
                
                #step 8.2 create a target sentence mask
                target_mask = transformer_model.get_target_mask(target_bos)
                
                #step 8.3 place the current output, encoder output and both masks into the decoder
                with torch.no_grad():
                    decoder_output,decoder_attention = transformer_model.decoder_stack(target_bos,encoder_output,
                                                                                    source_mask,target_mask)
                
                #step 8.4 get next output token prediction from decoder along with attention
                next_predicted_token = decoder_output.argmax(2)[:,-1].item()
                
                #step 8.5  add prediction to current output sentence prediction
                target_bos.append(next_predicted_token)
                
                #step 8.6 break if the prediction was an <eos> token
                if next_predicted_token == target_eos:
                    break
                i+=1 
            
            end = time()
            
            #step 9 convert the output sentence from indexes to tokens
            target_tokens = [self.vocab.itos[idx] for idx in target_bos]
            
            # if verbose is True:
            #     print(f'Time taken to translate {end - start} seconds')
            
            # step 10 return the output sentence (with the <sos> token removed) and the attention from the last layer
            return target_tokens[1:-1]        #,decoder_attention


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

    dm = Prd2ReactDataModule(config.filePath, batch_size=config.batch_size, splits=config.split)
    dm.prepare_data()
    dm.setup()

 

    source_pad_index = dm.source_vocab['<pad>']
    target_pad_index = dm.target_vocab['<pad>'] 

    print(source_pad_index)  

    encoder_stack = Encoder(input_dimension = len(dm.source_vocab), hidden_dimension = config.encoder_hid_dim,
                    number_encoder_layers = config.encoder_num_layers, num_attention_heads = config.encoder_nheads, 
                    pointwise_ff_dim = config.encoder_ff_dim, dropout = config.encoder_dropout)

    decoder_stack = Decoder(output_dimension = len(dm.target_vocab), hidden_dimension = config.decoder_hid_dim,
                    number_decoder_layers = config.decoder_num_layers, num_attention_heads = config.decoder_nheads, 
                    pointwise_ff_dim = config.decoder_ff_dim, dropout = config.decoder_dropout)


    transformer_model = Seq2SeqModel(encoder_stack, decoder_stack, source_pad_idx = dm.source_vocab['<pad>'],
                                                         target_pad_idx = dm.target_vocab['<pad>'])

    model = Prd2ReactTask(model = transformer_model, loss_fn = eval(config.loss_fn), optimizer=eval(config.optimizer), 
                                    metrics = eval(config.metrics), learning_rate = config.learning_rate, vocab = dm.target_vocab) 

    tb_logger = pl_loggers.TensorBoardLogger('logs/Transformer')


    trainer = flash.Trainer(max_epochs=config.epochs,gpus=0, logger=tb_logger)
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)
    

    # ckpt_file_path = '/home/bayeslabs/molFlash/molflash/retrosynthesis/Seq2Seq/Transformer/logs/Transformer/default/version_35/checkpoints/epoch=9-step=279.ckpt'
    # ckpt = torch.load(ckpt_file_path)

    # model.load_state_dict(ckpt["state_dict"])
    # print(model.predict('O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCN(C(=O)C2CC2)CC1'))
    
    


