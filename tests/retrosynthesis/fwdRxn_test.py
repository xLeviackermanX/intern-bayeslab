from omegaconf import OmegaConf
from argparse import ArgumentParser

from torchmetrics import Accuracy, F1
from torch import nn, Tensor, optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.model import Task
from flash.text.seq2seq.translation.metric import BLEUScore


from molflash.retrosynthesis.Seq2Seq.Transformer.fwdModel import React2ProdTask
from molflash.retrosynthesis.Seq2Seq.Transformer.fwdData import React2ProdDataModule
from molflash.models.Seq2Seq_transformer import Translation

if __name__=="__main__":


    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)
    config = configFile.config


    dm = React2ProdDataModule(config.filePath, batch_size=config.batch_size, split=config.split)
    dm.prepare_data()
    dm.setup()


    transformer_model = Translation(src_vocab_size=dm.reactant_vocab_size, tgt_vocab_size=dm.product_vocab_size, src_pad_index=dm.reactant_pad_index, tgt_pad_index=dm.product_pad_index)

    src_pad_index=dm.reactant_pad_index

    model = React2ProdTask(model = transformer_model, loss_fn = eval(config.loss_fn), optimizer=eval(config.optimizer), metrics = eval(config.metrics), src_pad_index=dm.reactant_pad_index,tgt_pad_index=dm.product_pad_index, tgt_vocab_size=dm.product_vocab_size) 
    tb_logger = pl_loggers.TensorBoardLogger('logs/FwdModel')


    trainer = flash.Trainer(max_epochs=config.epochs,gpus=1, logger=tb_logger)
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)