from omegaconf import OmegaConf
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
from pytorch_lightning import loggers as pl_loggers

import torch
from torch import nn, Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torchmetrics
from torchmetrics import Accuracy, F1

import flash
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess

from molflash.utils.preprocess import PreprocessingFunc
from molflash.models.fingerprintEncoder import InScopeFilterNet
from molflash.retrosynthesis.classification.data import PrdRxnDataModule, PrdRxnDataSource



class RetroTask(flash.Task):

        
    """ Task for classification.

    Args :
    model: Model used for the task
    num_classes: Number of classes to classify
    optimizer: Optimizer for training
    metrics: Metrics to compute training and evaluation
    learning_rate: Learning rate for training
    

    Returns: The Classification Task for the classification 
    """

    def __init__(self,
        model: nn.Module = None,
        learning_rate: float = 0.2,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None):
        
        optimizer = torch.optim.Adam

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self.model(x)

        output = {"y_hat": y_hat}
        output["y"] = y

        loss = 0
        loss += self.loss_fn(y_hat, y)

        metric = self.metrics(y_hat, y.int())
        logs = {}
        logs["loss"] = loss
        logs['metrics'] = metric
        output["loss"] = loss
        output['metrics'] = metric
        output["logs"] = logs
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



class ProdRxnPreprocess(Preprocess):

    """ The Preprocessing Class for the task
    
    Args:
    transform: The specific preprocessing function we need to call for the task       
    
    for example: fps_preprocess    
    """
    
    def __init__(self, transform = None):

        self.transform = transform

        super().__init__(data_sources={"numpy": PrdRxnDataSource()}, default_data_source="numpy")
    
    
    def pre_tensor_transform(self, sample: Any) -> Tuple:
        
        """ The function for preprocessing the data before converting into tensors


        Args: The data returned from load_data function from DataSource

        Result: The Processed data

        """
        x,y = sample
        x = self.transform(x)
        return (x,y)

        

    def to_tensor_transform(self, sample: Any) -> Tuple:
        
        """The function to convert the sample data into tensors"""
    
        x, y = sample
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.float)
        return (x, y)
    
    def collate_fn(self, batch: Any) -> Any:
        
        """ The collate function for the dataloaders"""

        data_list, label_list = [], []
        for _data, _label in batch:
            data_list.append(_data)
            label_list.append(_label)   
        return data_list, label_list

    
    def predict_to_tensor_transform(self, sample: Any) -> Tensor:

        """Function to transform predict data"""

        return torch.from_numpy(sample).float()


    def get_state_dict(self) -> Dict[str, Any]:
        return {}


    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get congiguration
    configFile = OmegaConf.load(args.config)
    config = configFile.config


    # create DataModule
    dm = PrdRxnDataModule.from_dataset(config.filePath, ProdRxnPreprocess(eval(config.preprocessing)), batch_size=config.batch_size, valid_split=config.splits[2],test_size=config.splits[1])

    
    # building a classification task using the model and the needed parameters
    model = RetroTask(eval(config.modelName)(), optimizer=optim.Adam, loss_fn=eval(config.loss_fn), metrics=eval(config.metrics))
    tb_logger = pl_loggers.TensorBoardLogger('logs/ReactionFeasibility')

    # buliding the trainer for the model
    trainer = flash.Trainer(max_epochs=config.epochs, gpus=1, progress_bar_refresh_rate=20, logger=tb_logger)
    trainer.fit(model, datamodule=dm)

    # Testing the model
    trainer.test(model, datamodule=dm)





