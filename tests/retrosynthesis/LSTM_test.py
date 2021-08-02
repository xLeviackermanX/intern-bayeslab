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

from molflash.retrosynthesis.Seq2Seq.LSTM.data import Prd2ReactDataModule
from molflash.retrosynthesis.Seq2Seq.LSTM.model import Prd2ReactTask
from molflash.models.Seq2Seq_retro import Seq2Seq,Encoder,Decoder

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