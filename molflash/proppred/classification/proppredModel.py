# All imports
from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
from pytorch_lightning.loggers import  TensorBoardLogger


# torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, F1
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# flash imports
import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.classification import ClassificationTask

# custom imports
from molflash.utils.emnPrepro import smile_to_graph
from molflash.models.emn import EMN
from molflash.proppred.emnData import emnDataModule
from molflash.models.gcn import GCN
from molflash.proppred.gcnData import gcnDataModule
from molflash.utils.preprocess import PreprocessingFunc

class CPPClassificationTask(flash.Task):
    """
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
    """

    def __init__(
            self, model: Optional[nn.Module] = None,
            loss_fn: Optional[Union[Callable, Mapping, Sequence]] = nn.CrossEntropyLoss(),
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
            learning_rate: float = 0.001,
            modelName: str = "EMN",
            logger: int = 0
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,

        )
        self.optimizer = optimizer
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.modelName = modelName
        if self.modelName == "EMN":
            self.model = model()
        elif self.modelName == "GCN":
            self.model = model

        if logger:
            self.logger = TensorBoardLogger("tb_logs", name = "model")


    def forward(self, x: Any) -> Any:
        pass

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        Basic step function
        """
        if self.modelName == "EMN":
            x, y = batch[:3], batch[3]
            y_hat = self.model(x)
        elif self.modelName == "GCN":
            x, y = batch[0], batch[0].y
            y_hat = self.model(x.x, x.edge_index)

        yh = y_hat > 0.5
        yh = yh.float()
        metric = self.metrics(yh, y.int())

        logs = {}
        output = {"y_hat": y_hat}
        output["y"] = x
        loss = self.loss_fn(y_hat, y)

        logs["loss"] = loss
        logs['metrics'] = metric
        output["metrics"] = metric
        output["loss"] = loss
        output["logs"] = logs

        return output

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        training loop
        """
        outputs = self.step(batch, batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_metric", outputs["metrics"])

        # if self.current_epoch == 0:
        #     smile = "CCCC"
        #     if self.modelName == "GCN":
        #         sample = PreprocessingFunc.smile_to_tensor(smile, 40)
        #         # print(sample)
        #         self.logger.experiment.add_graph(self.model(sample))
        return loss

    def training_epoch_end(self, outputs):

        # meant to get model graph
        if self.current_epoch == 0 and self.logger:
            smile = "CCCC"
            if self.modelName == "GCN":
                sample = PreprocessingFunc.smile_to_tensor(smile, 40)
            self.logger.experiment.add_graph(GCN(128, 256, 128, 512, 128, 1, 0.5), (sample.x, sample.edge_index))
        return

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch, batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metric", outputs["metrics"])
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.step(batch, batch_idx)
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


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)
    modelName = configFile.config.modelName
    filePath = configFile.config.filePath
    optimizer = configFile.config.optimizer
    learning_rate = configFile.config.learning_rate
    batch_size = configFile.config.batch_size
    splits = configFile.config.splits
    epochs = configFile.config.epochs
    gpus = configFile.config.gpus
    logger = configFile.config.logger
    loss_fn = nn.BCELoss()
    metrics = F1(num_classes=1)

    if modelName == "EMN":
        """Load and Create task for EMN Model"""
        print(f"model Used:{modelName}")
        dm = emnDataModule(filePath=filePath, transform=smile_to_graph, batch_size=batch_size, splits=splits)
        dm.prepare_data()
        dm.setup()
        model = CPPClassificationTask(model=eval(modelName), loss_fn=loss_fn, learning_rate=learning_rate,
                                      optimizer=eval(optimizer),
                                      metrics=metrics, modelName=modelName, logger = logger)
        torch.save(model.state_dict(), "/home/bayeslabs/molFlash/proppred/")
        tb_logger = pl_loggers.TensorBoardLogger('logs/EMN')

    elif modelName == "GCN":
        """Load and Create task for GCN Model"""
        print(f"model Used:{modelName}")
        dm = gcnDataModule(filepath=filePath, splits=[0.2, 0.1], batch_size=batch_size)
        dm.prepare_data()
        dm.setup()
        params = {'in_channel': 40,
                  'hid1': 128,
                  'hid2': 256,
                  'hid3': 128,
                  'lin1': 512,
                  'lin2': 128,
                  'out': 1,
                  'drop': 0.5,
                  }

        model = CPPClassificationTask(model=GCN(**params), loss_fn=loss_fn, learning_rate=learning_rate,
                                      optimizer=eval(optimizer),
                                      metrics=metrics, modelName=modelName)
        tb_logger = pl_loggers.TensorBoardLogger('logs/GCN')

    trainer = flash.Trainer(max_epochs=epochs, gpus=gpus, progress_bar_refresh_rate=20, logger=tb_logger)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

