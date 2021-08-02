
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
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, F1


import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask


#from molflash.retrosynthesis.classification.data import prdRxnDataModule, prdRxnDataSource   # gcnDataModule, gcnDataSource

from molflash.utils.emnPrepro import smile_to_graph
from molflash.retrosynthesis.classification.FWDRxnFesibility.FWDRxnScopeData import CPPDataModule
from molflash.models.gcn2 import FWDRxnGcn



class FWDRxnScopeTask(flash.Task):
    """
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
    """

    def __init__(
        self,model: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = nn.CrossEntropyLoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
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
        self.model = model

    def forward(self, x: Any) -> Any:
        pass
        
    def step(self, batch: Any, batch_idx: int) -> Any:
        x,y = batch, batch.y

        y_hat = self.model(x)
        yh = y_hat>0.5
        yh = yh.float()
        metric = self.metrics(yh, y.int())
        
        logs = {}
        output = {"y_hat": y_hat}
        output["y"] = x
        loss = self.loss_fn(y_hat,y)
        logs["loss"] = loss
        logs['metrics'] = metric
        output["metrics"] = metric

        output["loss"] = loss
        output["logs"] = logs

        return output


    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        # print(outputs["logs"])
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

    


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()
    # get congiguration
    configFile = OmegaConf.load(args.config)
    config = configFile.config
    dm = CPPDataModule(preprocess="get_graphfeatures",splits=config.splits, filepath=config.filePath)
    dm.prepare_data()
    dm.setup()

    model = FWDRxnScopeTask(model = eval(config.modelName)(**config.params), loss_fn=eval(config.loss_fn)(), learning_rate=config.learning_rate, optimizer=eval(config.optimizer),metrics=eval(config.metrics)(num_classes=1))
    tb_logger = pl_loggers.TensorBoardLogger('logs/FWDRxnGcn')

    trainer = flash.Trainer(max_epochs=config.epochs,gpus=config.gpus, progress_bar_refresh_rate=20, logger=tb_logger)

    trainer.fit(model, datamodule=dm)
    trainer.test(model,datamodule=dm)

