import sys
import argparse
import math

import torch
import pandas as pd
from rdkit import Chem
from functools import partial
from torch.multiprocessing import Pool
from utils.preprocess import find_clusters, extract_subgraph
from utils.preprocess import get_scoring_function

from multiprocessing import Pool


from MCTS import mcts

class task:
    #call MCTS code

    def substructurefn(self):
        work_func = partial(mcts, scoring_function=scoring_function,
                            n_rollout=self.rollout,
                            # min_atoms=self.min_atoms,
                            max_atoms=self.max_atoms,
                            min_threshold=self.min_threshold,
                            max_threshold=self.max_threshold)

        results = self.pool.map(work_func, data)

        for orig_smiles, rationales in results:
            # print(orig_smiles)
            rationales = sorted(rationales, key=lambda x: len(x.atoms))
            for x in rationales[:self.args.ncand]:
                if x.smiles not in rset:
                    gen_rational = {'smiles': orig_smiles, 'rationales': x.smiles, 'num_atoms': len(x.atoms),
                                    'reward': x.P}
                    self.mcts_rationales = self.mcts_rationales.append(gen_rational, ignore_index=True)
                    rset.add(x.smiles)
        self.mcts_rationales.to_csv("Mcts_rationales.csv")
        return self.mcts_rationales


class CustomDataModule(DataModule):
    def get_data(self,args):
        return pd.read_csv(args.file)

    def from_dataset(self):
        """
        preprocessing required if any
        :return:
        """


if __name__=="__main__":
    #config file load and call functions
