from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
from molflash.utils.preprocess import collate_fn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from molflash.generator.utils.Seq2Seq_preprocess import input_data
from molflash.utils.utils import split_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from molflash.generator.utils.preprocess import build_vocab, tokenize, generator_preprocess, smile_preprocess


class Seq2SeqDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str = None, batch_size: Optional[int] = 2, splits: Optional[list] = [0.8, 0.1, 0.1]):
        super().__init__()

        if file_path is None:
            raise ValueError("No Data Directory Provided.")
        
        self.splits = splits
        self.file_path = file_path
        self.batch_size = batch_size
        self.source_vocab = None
        self.tokenizer = tokenize

    def prepare_data(self) -> None:
        self.source_vocab = build_vocab(self.file_path, self.tokenizer)
        df = pd.read_csv(self.file_path)
        self.data = list(df['smiles'])

        # self.data = generator_preprocess(self.file_path, self.source_vocab, self.tokenizer)

    def setup(self, stage: Optional[str] = None):
        train_len = int(self.splits[0]*len(self.data))
        test_len = int(self.splits[1]*len(self.data))
        val_len = len(self.data)-train_len-test_len
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)

    def collate_fn(self, data_batch):
        prod_batch=[torch.cat([torch.tensor([self.source_vocab['<bos>']]), smile_preprocess(prod_item,self.source_vocab, self.tokenizer), torch.tensor([self.source_vocab['<eos>']])], dim=0) for prod_item in data_batch]
        prod_batch = pad_sequence(prod_batch, padding_value=self.source_vocab['<pad>'])
        return prod_batch
        



