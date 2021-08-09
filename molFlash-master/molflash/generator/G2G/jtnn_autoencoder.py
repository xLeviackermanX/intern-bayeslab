import flash
from torch_geometric.nn.models import GAE
from flash.core.model import Task
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Mapping, Sequence
from torch import nn
import torch
import torchmetrics
import pytorch_lightning as pl
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Postprocess, Preprocess, Serializer, SerializerMapping
from flash.core.registry import FlashRegistry
from flash.core.schedulers import _SCHEDULERS_REGISTRY
from flash.core.utilities.apply_func import get_callable_dict
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from molflash.models.jtnn.jtmpn import JTMPN
from molflash.models.jtnn.mpn import MPN
TENSOR = torch.tensor
class JTNNTask(Task):
    """General Task for Graph2Graph.
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
            vocab, hidden_size: int, latent_size: int, depthT, depthG,
            model: nn.Module,
            loss_fn_1: Optional[Union[Callable, Mapping, Sequence]] = None,
            loss_fn_2: Optional[Union[Callable, Mapping, Sequence]] = None,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
            learning_rate: float = 5e-5
    ):
        optimizer = torch.optim.Adam
        super().__init__(
            loss_fn=nn.BCELoss(),
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.optimizer = optimizer
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.model = model(vocab, hidden_size, latent_size, depthT, depthG)
        # self.loss_fn = loss_fn

    # def forward(self, x, edge_index) -> Any:
    #     """
    #     encoder step
    #     returns -> z and kl_loss
    #     """
    #     z = self.encoder(x, edge_index)
    #     mu = torch.mean(z)
    #     logvar = torch.log(torch.var(z))
    #     kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum().mean(0)
    #     return z, kl_loss

    # def dec_forward(self, z) -> Any:
    #     """
    #     decoder step
    #     """
    #     return self.decoder(z)

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        Makes a forward pass through the encoder, then the decoder.
        After the Decoder step, various losses are calculated and then added
        """
        loss, kl_div, wacc, tacc, sacc = self.model(batch, beta = 0.0)
        # x, edge_index, edge_features = batch[0].x, batch[0].edge_index, batch[0].edge_feature


        outputs = {}

        logs = {}
        logs["loss"] = loss
        outputs["logs"] = logs
        outputs["loss"] = loss
        outputs["kl_div"] = kl_div
        outputs["wacc"] = wacc
        outputs["tacc"] = tacc
        outputs["saac"] = sacc

        return outputs

    # def get_atoms(self, data: TENSOR) -> List:
    #     symbol_set = ['C', 'N', 'O', 'S', 'F', 'H', 'P', 'Cl', 'Br', 'K', 'Mg', 'Si']
    #     atoms_list = []
    #     for i in range(len(data)):
    #         data[i]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.step(batch, batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
    #     outputs = self.step(batch, batch_idx)
    #     loss = outputs["loss"]
    #     self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
    #     return loss
    #
    # def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
    #     outputs = self.step(batch, batch_idx)
    #     loss = outputs["loss"]
    #     self.log_dict({f"test_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
    #     return loss

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = self.optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer_kwargs["lr"] = self.learning_rate
            optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_kwargs)
        if self.scheduler:
            return [optimizer], [self._instantiate_scheduler(optimizer)]
        return optimizer

