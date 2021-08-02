from omegaconf import OmegaConf
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch import nn, Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import flash
from flash.core.model import Task
from molflash.generator.Seq2Seq.s2sdata import Seq2SeqDataModule
from molflash.models.seq2seq_new import Seq2Seq, Encoder, Decoder

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
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        hidden_size: int = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = nn.CrossEntropyLoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 0.001,
        tgt_vocab_size: int = None,
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
        self.optimizer = optimizer
        self.q_mu = nn.Linear(hidden_size, hidden_size)
        self.q_logvar = nn.Linear(hidden_size, hidden_size)
        self.tgt_vocab_size = tgt_vocab_size
        self.metrics = metrics

    def forward(self, x: Any) -> Any:
        """pass inputs to encoder
        --->calculate Encoder outputs
        --->Calculate loss
        --->return hidden, cell, loss"""
        hidden, cell = self.encoder(x)
        h = torch.mean(hidden,1)
        c = torch.mean(hidden,1)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        mu1, logvar1 = self.q_mu(c), self.q_logvar(c)
        eps1 = torch.randn_like(mu1)
        c = mu1 + (logvar1 / 2).exp() * eps1

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z.unsqueeze(1),c.unsqueeze(1),kl_loss

    
    def dec_forward(self, x: Any, hidden: Any, cell: Any) -> Any:
        """pass x,Encoder hidden,cell to Decoder
        --->return decoder outputs"""
        return self.decoder(x, hidden, cell)


    def step(self, batch: Any, batch_idx: int) -> Any:
        '''batch
        --->pass batch through forward
        --->pass forward outputs through dec_forward
        ---> calculate loss
        ---> return outputs
        '''
        
        x = batch
        source = x
        loss = 0
        batch_size = source.shape[0]
        target_len = source.shape[1]
        target_vocab_size = self.tgt_vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, cell, kl_loss = self.forward(source)
       
        hidden = torch.cat([hidden]*batch_size, dim=1)
        cell = torch.cat([cell]*batch_size, dim=1)

        # Grab the first input to the Decoder which will be <SOS> token
        x_dec = source[:, 0].unsqueeze(0)
        
        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.dec_forward(x_dec, hidden, cell)
    
            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            x_dec = best_guess.unsqueeze(0)

        y_hat = outputs

        metric = self.metrics(y_hat.argmax(-1).permute(1,0), x.cpu())
        
        y_hat = y_hat.reshape(x.shape[0], -1, x.shape[1])

        logs = {}
        output = {"y_hat": y_hat}
        output["y"] = x
        loss += self.loss_fn(y_hat, x.cpu())
        logs["loss"] = loss
        output["metrics"] = metric

        output["loss"] = loss+kl_loss
        output["logs"] = logs

        return output

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"train_{k}": v for k, v in outputs["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_metric", outputs["metrics"])
        return loss

    def common_step(self, prefix: str, batch: Any) -> Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        outputs = self.step(batch,batch_idx)
        loss = outputs["loss"]
        self.log_dict({f"val_{k}": v for k, v in outputs["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metric", outputs["metrics"])

        return loss


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
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

    # get congiguration
    configFile = OmegaConf.load(args.config)

    config = configFile.config
    dm = Seq2SeqDataModule(config.filePath, batch_size=config.batch_size, splits=config.splits)
    dm.prepare_data()
    dm.setup()


    input_size = len(dm.vocab)

    encoder_net = Encoder(input_size=input_size, embedding_size=config.enc_emb_size, hidden_size=config.enc_hid_size, num_layers=config.enc_n_layers, p=config.enc_drop)
    decoder_net = Decoder(input_size=input_size, embedding_size=config.dec_emb_size, hidden_size=config.dec_hid_size, output_size=input_size, num_layers=config.dec_n_layers, p=config.dec_drop)

    model = Seq2SeqGenTask(encoder=encoder_net, decoder=decoder_net, hidden_size=config.hidden_size,
                           learning_rate=config.lr, loss_fn=eval(config.loss_fn), optimizer=eval(config.optimizer),
                           metrics=eval(config.metrics)(num_classes=input_size, mdmc_average='samplewise'),
                           tgt_vocab_size=input_size)
    tb_logger = pl_loggers.TensorBoardLogger('logs/Generation')

    trainer = flash.Trainer(max_epochs=config.epochs, gpus=config.gpus, progress_bar_refresh_rate=20, logger=tb_logger)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

