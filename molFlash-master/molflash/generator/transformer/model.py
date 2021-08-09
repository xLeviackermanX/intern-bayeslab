import os
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type

import torch
from torch import nn, Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import flash
from flash.core.model import Task

from molflash.generator.transformer.data import GENDataModule
from molflash.generator.utils.preprocess import Dataset, sample
from molflash.models.transformerGEN.encode_decode.model import EncoderDecoder


CUDA_LAUNCH_BLOCKING = 1


class Seq2SeqGenTask(Task):
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
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = nn.CrossEntropyLoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
        vocab = None,
    ):

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
           
        )
        self.model = model
        self.optimizer=optimizer
        self.metrics=metrics
        self.learning_rate=learning_rate
        self.loss_fn = loss_fn
        self.vocab = vocab

    

    def step(self, batch: Any, batch_idx: int) -> Any:
        '''batch
        --->pass batch through forward
        --->pass forward outputs through dec_forward
        ---> calculate loss
        ---> return outputs
        '''
         
        src, source_length, trg, src_mask, trg_mask, _, _ = batch

        trg_y = trg[:, 1:] 

        trg = trg[:, :-1]
        out = self.model.forward(src, trg, src_mask, trg_mask)
        out = self.model.generator(out)

        loss = 0    
        y_hat = out.permute(0,2,1)

        logs = {}
        output  = {}

        output = {"y_hat": y_hat}
        # output["y"] = x
        loss += self.loss_fn(y_hat, trg_y)

        logs["loss"] = loss
        # output["metrics"] = metric

        output["loss"] = loss
        output["logs"] = logs
        return output

    def common_test_step(self, batch: Any, model) -> Any:
        '''batch
        --->pass batch through forward
        --->pass forward outputs through dec_forward
        ---> calculate loss
        ---> return outputs
        '''
        # model = self.model.load_from_file()
        model = model
        model.eval()
        df_list = []
        sampled_smiles_list = []

        src, source_length, _, src_mask, _, _, df = batch
        max_len = 75
        num_samples = 10
        # decode_type = 'greedy'
        decode_type = 'multinomial'

        smiles= sample(model, src, src_mask, source_length, decode_type, self.vocab, num_samples = num_samples, max_len = max_len)
                                                                   

        df_list.append(df)
        sampled_smiles_list.extend(smiles)

        data_sorted = pd.concat(df_list)
        sampled_smiles_list = np.array(sampled_smiles_list)

        for i in range(num_samples):
            data_sorted['Predicted_smi_{}'.format(i + 1)] = sampled_smiles_list[:, i]

        # parent_path = get_parent_dir(self.filePath)
        parent_path = '/home/bayeslabs/molFlash/molflash/generator/transformer/'
        result_path = os.path.join(parent_path, "generated_molecules.csv")
        data_sorted.to_csv(result_path, index=False, mode='a')


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_metric", outputs["metrics"])
        return loss

    def common_step(self, prefix: str, batch: Any) -> Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_metric", outputs["metrics"])

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

    dm = GENDataModule(config.filePath, batch_size=config.batch_size, collate_fn = Dataset.collate_fn)
    dm.prepare_data()
    dm.setup()


    model_net = EncoderDecoder.make_model(len(dm.vocab), len(dm.vocab), config.num_layers, config.hidden_dim, config.feedforward_dim, config.no_heads, config.dropout)


    model = Seq2SeqGenTask(model = model_net,
                           learning_rate=config.lr, loss_fn= eval(config.loss_fn), optimizer=eval(config.optimizer), vocab=dm.vocab)
    tb_logger = pl_loggers.TensorBoardLogger('logs/Generation')

    trainer = flash.Trainer(max_epochs=config.epochs, gpus=config.gpus, progress_bar_refresh_rate=20, logger=tb_logger)

    trainer.fit(model, datamodule=dm)

    # # trainer.test(model, datamodule=dm)

    # ckpt_file_path = '/home/bayeslabs/molFlash/molflash/generator/transformer/logs/Generation/default/version_0/checkpoints/epoch=2-step=7538.ckpt'
    # ckpt = torch.load(ckpt_file_path)

    # model.load_state_dict(ckpt["state_dict"])

    # dataloader = dm.test_dataloader()
    # for batch in dataloader:
    #     print(batch)
    #     model.common_test_step(batch, model.model)








