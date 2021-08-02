# IMPORTS
import pandas as pd

PD = pd.DataFrame


class Dataset(torch.utils.data.Dataset):  # subclass torch.utils.data.Dataset

    def __init__(
            self,
            smiles: Any,
    ):
        self.smiles = smiles

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        smile = self.smiles["smiles"][index]


        sample = (smile)

        return sample

    def __len__(self):
        return len(self.smiles)


class NumpyDataSource(flash.DataSource[Tuple[ND, ND]]):

    def load_data(self, data: Tuple[ND, ND], dataset: None) -> List[Dict[str, Any]]:
        # x, y = data
        x = data

        ds = Dataset(x)
        return ds


class GCNPreprocess(flash.Preprocess):
    def __init__(self):
        super().__init__(data_sources={"tensors": NumpyDataSource()}, default_data_source="tensors")

    # the smiles are processed to get

    def pre_tensor_transform(self, smiles: Any) -> Any:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)

        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol

        return mol_tree

    # def collate(self, samples: Sequence) -> Any:
    #     return samples

    def get_state_dict(self) -> Dict[str, Any]:
        # TODO -> fill this code
        return {}

    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        # TODO -> fill this code
        return {}


class DataModule():  # subclass flash datamodule

    # opens the file and loads it in pandas dataframe
    @classmethod
    def get_data(cls, path) -> PD:
        return pd.read_csv(path)

    # remove the row having smiles that are incorrect or their
    # size is smaller than the threshold
    @classmethod
    def clean_data(cls, data, threshold): ...


    # this function will be called
    # and return the datamodule
    @classmethod
    def from_dataset(cls, path, preprocess: Preprocess, batch_size: int = 1, num_workers: int = 0): ...





