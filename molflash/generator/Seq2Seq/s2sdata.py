from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from molflash.generator.utils.Seq2Seq_preprocess import input_data
from molflash.utils.utils import split_data


class Seq2SeqDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str = None, batch_size: Optional[int] = 2, splits: Optional[list] = [0.8, 0.1, 0.1]):
        super().__init__()

        if file_path is None:
            raise ValueError("No Data Directory Provided.")
        
        self.splits = splits
        self.file_path = file_path
        self.batch_size = batch_size
        self.vocab = None

    def prepare_data(self) -> None:
        self.data, self.vocab = input_data(self.file_path)

    def setup(self, stage: Optional[str] = None):
        train_len, test_len, val_len = split_data(len(self.data), self.splits)
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4)
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)









