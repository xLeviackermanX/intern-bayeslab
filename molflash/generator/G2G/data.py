# IMPORTS
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from molflash.utils.preprocess import PreprocessingFunc
import pytorch_lightning as pl
from molflash.utils.utils import split_data
import os
import pickle
import random
import torch
from molflash.models.jtnn import *
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from multiprocessing import Pool
# from molflash.utils.preprocess import tensorize
from molflash.models.jtnn.vocab import Vocab

PD = pd.DataFrame
ND = np.ndarray


class EncoderDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str = None, batch_size: Optional[int] = 1, num_workers: Optional[int] = 4, splits: Optional[list] = [0.8, 0.1, 0.1]):
        super().__init__()

        if file_path is None:
            raise ValueError("No Data Directory Provided.")

        self.data = None
        self.file_path = file_path
        self.batch_size = batch_size
        self.splits = splits
        self.num_workers = num_workers
        self.train_data, self.val_data, self.test_data = None, None, None

    def get_data(self, path) -> PD:
        return pd.read_csv(path, usecols=["smiles"])

    def clean_data(self, data: PD, threshold=40) -> PD:
        for index, row in data.iterrows():
            try:
                Chem.MolFromSmiles(data["smiles"][index])
            except:
                data.drop(index, inplace=True)
        data["num_atoms"] = [Chem.MolFromSmiles(smile).GetNumAtoms() for smile in data['smiles']]
        data = data[data["num_atoms"] <= threshold]
        data = data.drop("num_atoms", axis=1)
        return data

    def prepare_data(self) -> None:
        self.data = self.get_data(self.file_path)
        self.data = self.clean_data(self.data)
        self.data = PreprocessingFunc.smiles_features(self.data)


    def setup(self, stage: Optional[str] = None):
        train_len, test_len, val_len = split_data(len(self.data), self.splits)
        self.train_data, self.val_data, self.test_data = random_split(self.data, [train_len, test_len, val_len])


    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


"""
JTNN
"""


class JTNNDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str = None, batch_size: int = 32, num_splits: int = 100, num_workers: int = 8,
                 shuffle=True, assm=True, replicate=None):
        super().__init__()
        self.file_path = file_path
        self.data_files = None
        self.batch_size = batch_size
        self.vocab = list(pd.read_csv("/home/trinity/github-repos/new/molFlash/molflash/utils/vocab.csv")["smiles"])
        self.vocab = Vocab(self.vocab)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.num_splits = num_splits
        self.data = None
        self.dataset = None
        # if replicate is not None:  # expand is int
        #     self.data_files = self.data_files * replicate

    def get_data(self, path) -> PD:
        return pd.read_csv(path, usecols=["smiles"], nrows = 2000)

    def clean_data(self, data: PD, threshold=40) -> PD:
        for index, row in data.iterrows():
            try:
                Chem.MolFromSmiles(data["smiles"][index])
            except:
                data.drop(index, inplace=True)
        data["num_atoms"] = [Chem.MolFromSmiles(smile).GetNumAtoms() for smile in data['smiles']]
        data = data[data["num_atoms"] <= threshold]
        data = data.drop("num_atoms", axis=1)
        return data

    def prepare_data(self) -> None:
        self.data = self.get_data(self.file_path)
        self.data = self.clean_data(self.data)

        pool = Pool(self.num_workers)
        all_data = pool.map(tensorize, list(self.data["smiles"]))
        le = (len(all_data) + self.num_splits - 1) // self.num_splits
        os.mkdir("/home/trinity/github-repos/new/molFlash/molflash/generator/G2G/tensors/")

        for split_id in range(self.num_splits):
            st = split_id * le
            sub_data = all_data[st: st + le]

            with open('/home/trinity/github-repos/new/molFlash/molflash/generator/G2G/tensors/tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

        self.data_files = [fn for fn in os.listdir("/home/trinity/github-repos/new/molFlash/molflash/generator/G2G/tensors/")]
        for fn in self.data_files:
            fn = os.path.join("/home/trinity/github-repos/new/molFlash/molflash/generator/G2G/tensors/", fn)

            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            # if len(batches[-1]) < self.batch_size:
            #     batches.pop()

        self.dataset = MolTreeDataset(batches, self.vocab, self.assm)

    def setup(self, stage: Optional[str] = None):
        self.data = self.dataset

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return TorchDataLoader(self.data, batch_size=1, num_workers=self.num_workers, collate_fn=lambda x:x[0])

    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
    #     return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
    #
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
    #     return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _tensorize(self.data[idx], self.vocab, assm=self.assm)


def _tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:

            # Leaf node's attachment is determined by neighboring node's attachment

            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1