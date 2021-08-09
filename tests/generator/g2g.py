import argparse
import torch
from torch import nn
import flash
from omegaconf import OmegaConf
from molflash.models import *
from molflash.generator.G2G.data import EncoderDataModule
from molflash.generator.G2G.autoencoder import G2GTask
from molflash.generator.G2G.data import EncoderDataModule

if __name__ == "__main__":
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get configuration
    configFile = OmegaConf.load(args.config)
    encoder = eval(configFile.config.encoder)()
    decoder = eval(configFile.config.decoder)()
    file_path = configFile.config.filePath
    epochs = configFile.config.epochs
    loss_fn_1 = eval(configFile.config.loss_fn_1)
    loss_fn_2 = eval(configFile.config.loss_fn_2)
    optimizer = configFile.config.optimizer
    gpus = configFile.config.gpus
    batch_size = configFile.config.batch_size
    lr = configFile.config.lr
    splits = configFile.config.splits

    # model = G2GTask(encoder=SGEncoder(), decoder=BasicDecoder(), loss_fn_1=nn.BCELoss(),
    #                 loss_fn_2=nn.MSELoss(), optimizer=torch.optim.Adam)
    model = G2GTask(encoder=encoder, decoder=decoder, loss_fn_1=loss_fn_1, loss_fn_2=loss_fn_2,
                    optimizer=optimizer)

    dm = EncoderDataModule(file_path=file_path, batch_size=batch_size, splits=splits)

    trainer = flash.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, gpus=gpus)
    trainer.fit(model, dm)


