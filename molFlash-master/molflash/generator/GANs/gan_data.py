#flash imports
#other imports


#preprocessing of smile

import pandas as pd
import rdkit
from rdkit import Chem
from torch.utils.data import Dataset
import itertools
import networkx as nx



class Vocab(object):
    """
    for mapping atoms
    """
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)




class SubgraphDataset(Dataset):

    def enum_root(self,smiles, num_decode):
        """
        :param smiles:  input smile for generating new ones
        :param num_decode: how many need to generated
        :return: roots based on number of rationales to be generated for given smile
        """
        mol = Chem.MolFromSmiles(smiles)
        roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
        self.outputs = []
        for perm_roots in itertools.permutations(roots):
            if len(self.outputs) >= num_decode: break
            mol = Chem.MolFromSmiles(smiles)
            for i, a in enumerate(perm_roots):
                mol.GetAtomWithIdx(a).SetAtomMapNum(i + 1)
            self.outputs.append(Chem.MolToSmiles(mol))

        while len(self.outputs) < num_decode:
            self.outputs = self.outputs + self.outputs
        return self.outputs[:num_decode]

    def __init__(self, data, avocab, batch_size, num_decode):
        data = [x for smiles in data for x in self.enum_root(smiles, num_decode)]
        self.batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def getbatches(self):
        return len(self.batches)

    def getbatchitem(self, idx):
        return self.batches[idx]


class MolGraph(object):
    """
    Generate the Molecular Graph
    """

    def get_mol(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: Chem.Kekulize(mol)
        return mol

    BOND_LIST = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
    MAX_POS = 40

    def __init__(self, smiles, init_atoms, root_atoms=None, shuffle_roots=True):
        self.smiles = smiles
        self.mol = self.get_mol(smiles)
        self.mol_graph = self.build_mol_graph()
        self.init_atoms = set(init_atoms)
        self.root_atoms = self.get_root_atoms() if root_atoms is None else root_atoms
        if len(self.root_atoms) > 0:
            if shuffle_roots: random.shuffle(self.root_atoms)
            self.order = self.get_bfs_order()

    def debug(self):
        for atom in self.mol.GetAtoms():
            if atom.GetIdx() in self.init_atoms:
                atom.SetAtomMapNum(atom.GetIdx())

    def get_root_atoms(self):
        roots = []
        for idx in self.init_atoms:
            atom = self.mol.GetAtomWithIdx(idx)
            bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in self.init_atoms]
            if len(bad_neis) > 0:
                roots.append(idx)
        return roots

    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index(bond.GetBondType())
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph




class MoleculeDataset(Dataset,MolGraph):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        init_smiles, final_smiles = zip(*self.batches[idx])
        init_batch = [Chem.MolFromSmiles(x) for x in init_smiles]
        mol_batch = [Chem.MolFromSmiles(x) for x in final_smiles]
        init_atoms = [mol.GetSubstructMatch(x) for mol, x in zip(mol_batch, init_batch)]
        mol_batch = [MolGraph(x, atoms) for x, atoms in zip(final_smiles, init_atoms)]
        mol_batch = [x for x in mol_batch if len(x.root_atoms) > 0]
        if len(mol_batch) < len(self.batches[idx]):
            num = len(self.batches[idx]) - len(mol_batch)
            print("MoleculeDataset: %d graph removed" % (num,))
        return MolGraph.tensorize(mol_batch, self.avocab) if len(mol_batch) > 0 else None



class PreprocessingFunc(Vocab,SubgraphDataset,MoleculeDataset):
    """
    Main class For all Dataset Preprocess Operations
    Vocab: for mapping
    SubgraphDataset: dataset object based on num decode
    """

    def __init__(self):
        """
        Initialize common atoms used for vmapping
        """
        self.COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0),
                    ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0),
                    ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1),
                    ('P', 0),
                    ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0),
                    ('Si', -1)]



    def vocabfn(self):
        """
        call the class Vocab for mapping with common atoms and values
        :return: Vocab object with mappings
        """
        return Vocab(self.COMMON_ATOMS)


    def subgraphdatasetfn(self, data, avocab, batch_size, num_decode):
        """

        :param data: rationale data
        :param avocab: Vocab ojbect
        :param batch_size: data batch size
        :param num_decode: number of rationales for each rationale
        :return: Subgraphdataset Object in batches
        """
        return SubgraphDataset(data, avocab, batch_size, num_decode)



    def moleculedatasetfn(self,moleculess,vocab,numdecode):
        return MoleculeDataset(moleculess, vocab, numdecode)



    def compfunc(self,modelattr=[]):
        """

        :return: setting up the property constraints
        """

        self.property_range = [(property['min_threshold'], property['max_threshold']) for property in model_attr]
        self.num_property = len(self.property_range)
        num_p = len(x)
        if num_p == 1:
            y = lambda x: self.property_range[0][0] <= x[0] <= self.property_range[0][1]
            return y(x)
        elif num_p == 2:
            y = lambda x: self.property_range[0][0] <= x[0] <= self.property_range[0][1] and \
                          self.property_range[1][0] <= x[1] <= self.property_range[1][1]
            return y(x)
        elif num_p == 3:
            y = lambda x: self.property_range[0][0] <= x[0] <= self.property_range[0][1] and \
                          self.property_range[1][0] <= x[1] <= self.property_range[1][1] and \
                          self.property_range[2][0] <= x[1] <= self.property_range[2][1]
            return y(x)
        elif num_p == 4:
            y = lambda x: self.property_range[0][0] <= x[0] <= self.property_range[0][1] and \
                          self.property_range[1][0] <= x[1] <= self.property_range[1][1] and \
                          self.property_range[2][0] <= x[2] <= self.property_range[2][1] and \
                          self.property_range[3][0] <= x[3] <= self.property_range[3][1]
            return y(x)
        elif num_p == 5:
            y = lambda x: self.property_range[0][0] <= x[0] <= self.property_range[0][1] and \
                          self.property_range[1][0] <= x[1] <= self.property_range[1][1] and \
                          self.property_range[2][0] <= x[2] <= self.property_range[2][1] and \
                          self.property_range[3][0] <= x[3] <= self.property_range[3][1] and \
                          self.property_range[4][0] <= x[4] <= self.property_range[4][1]
            return y(x)
        else:
            print("maximum no of filter property constrain must be less than 5")
            exit()




    def unique_rationales(smiles_list):
        """
         List of smiles as input
        :return: unique rationales from a given list
        """
        visited = set()
        unique = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            root_atoms = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    root_atoms += 1
                    atom.SetAtomMapNum(1)

            smiles = Chem.MolToSmiles(mol)
            if smiles not in visited and root_atoms > 0:
                visited.add(smiles)
                unique.append(smiles)
        return unique


    def rem_invalidsmiles(self,rationale_list):
        """

        :param rationale_list: list of smiles
        :return: rationales  list after invalids removal
        """
        new_rationales=[]
        for rationale in rationales:
            try:
                Chem.MolFromSmiles(rationale)
                new_rationales.append(rationale)
            except:
                continue
        return new_rationales


    def get_scoring_function(self,prop_name=None, prop_type=None, model_ckpt=None):
        """

        for neural net model
        :param prop_name: 'bbb','logd...'
        :param prop_type: "regression","classification"
        :param model_ckpt: "model_ckpt file"
        :return: returns a scoring function by name
        """

        if prop_name == 'qed':
            return qed_func()
        elif prop_name == 'sa':
            return sa_func()
        elif prop_name and prop_type and model_ckpt:
            return load_model_ckpt(prop_name=prop_name, model_type=prop_type, model_ckpt=model_ckpt)
        else:
            print("add valid model")


    def remove_order(s):
        """

        :return: remove the order of the smile by replacement
        """
        for x in range(15):
            s = s.replace(":%d]" % (x,), ":1]")
        return s




