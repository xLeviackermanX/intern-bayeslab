from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type


import random
import math
import time
import io
from os import sep
from numpy.core.fromnumeric import trace

import json
import re
import traceback


import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split, DataLoader

import torch_geometric
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Data

from collections import Counter

import traceback
FPS =  AllChem.GetMorganFingerprintAsBitVect
ND = np.array

random.seed(40)


class PrepDataset(Dataset):

        def __init__(self, list_ips, labels):

            self.list_ips = list_ips
            self.labels = labels

        def __getitem__(self, idx): 

            input = self.list_ips[idx]
            target = self.labels[idx]

            sample = (input, target)

            return sample

        def __len__(self):

            return len(self.list_ips)
    

class GeometricDataset(Dataset):
    
    def __init__(self, features, transform = None):

        self.features = features
        self.transform = transform
        
    def __len__(self):

        return (len(self.features))
    
    def __getitem__(self, index):
        
        sample = self.features[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class RxnDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        list_fps: Any,
        labels:int
    ):

        self.list_fps = list_fps

        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        fps = self.list_fps[index]
        label = self.labels[index]
        sample = (fps, label)

        return sample
    
    def __len__(self):
        return len(self.list_fps)


def collate_fn(batch):

    inp = [smile for smile,yy in batch]
    yy = torch.LongTensor([yy for smile,yy in batch])
    Xs,Ys = PreprocessingFunc.fps_preprocess(inp,yy)

    return Xs,Ys



class PreprocessingFunc:

    def __init__(self):

        self.BONDTYPE_TO_INT = defaultdict(
            lambda: 0,
            {
                BondType.SINGLE: 0,
                BondType.DOUBLE: 1,
                BondType.TRIPLE: 2,
                BondType.AROMATIC: 3
            }
        )


    @classmethod
    def one_of_k_encoding(cls, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


    @classmethod
    def one_of_k_encoding_unk(cls,x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


    @classmethod
    def atom_feature(cls, atom):
        symbol_set = ['C', 'N', 'O', 'S', 'F', 'H', 'P', 'Cl', 'Br', 'K', 'Mg', 'Si']  # 12
        # degree_set = [0, 1, 2, 3, 4, 5]  # 6
        num_hydrogens_set = [0, 1, 2, 3, 4]  # 5
        valency_set = [0, 1, 2, 3, 4, 5]  # 6
        formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]  # 7
        hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2]  # 5
        number_radical_e_list = [0, 1, 2]  # 3
        # chirality = ['R', 'S'] #1
        # Aromatic              #1
        mol_wt = [12.011, 14.007, 15.99, 32.065, 18.998, 1.007, 30.97, 35.453, 79.904, 39.098, 24.305, 28.085]
        m = np.mean(mol_wt)
        st_d = np.std(mol_wt)
        std_mol_wt = [(i - m) / st_d for i in mol_wt]

        return np.array(list(np.multiply(cls.one_of_k_encoding(atom.GetSymbol(), symbol_set), std_mol_wt)) +  ####12
                        # cls.one_of_k_encoding(atom.GetDegree(), degree_set) +
                        cls.one_of_k_encoding(atom.GetTotalNumHs(), num_hydrogens_set) +  ######5
                        cls.one_of_k_encoding(atom.GetImplicitValence(), valency_set) +  ######6
                        cls.one_of_k_encoding(atom.GetFormalCharge(), formal_charge_list) +  ######7
                        cls.one_of_k_encoding(atom.GetHybridization(), hybridization_list) +  ######5
                        cls.one_of_k_encoding(atom.GetNumRadicalElectrons(), number_radical_e_list) +  ######3
                        [atom.GetIsAromatic()] + [atom.HasProp('_ChiralityPossible')]).astype('float')  ######1+1


    @classmethod
    def graph_representation(cls, mol, max_atoms):
        adj = np.zeros((max_atoms, max_atoms))
        atom_features = np.zeros((max_atoms, 40))
        num_atoms = mol.GetNumAtoms()
        adj[0:num_atoms, 0:num_atoms] = Chem.rdmolops.GetAdjacencyMatrix(mol)
        edge0 = []
        edge1 = []
        for i, l in enumerate(adj):
            for j, k in enumerate(l):
                if (k == 1):
                    edge0.append(i)
                    edge1.append(j)
        edge_idx = [edge0, edge1]
        features_tmp = []
        for atom in mol.GetAtoms():
            features_tmp.append(cls.atom_feature(atom))
        atom_features[0:num_atoms, 0:40] = np.array(features_tmp)
        return edge_idx, atom_features


    @classmethod
    def smile_to_tensor(cls, smile, mol_length):
        mol = Chem.MolFromSmiles(smile)
        edge_index, x = cls.graph_representation(mol, mol_length)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        return data


    @classmethod
    def all_smile_to_tensor(cls, smiles, labels, threshold=40):
        graph_data = []
        for smile, label in zip(smiles, labels):
            try:
                if Chem.MolFromSmiles(smile).GetNumAtoms() <= threshold:
                    data = cls.smile_to_tensor(smile, threshold)
                    label = torch.tensor(label, dtype=torch.float).view(1, -1)
                    data.y = label
                    graph_data.append(data)
            except:
                traceback.print_exc()
                pass
        return graph_data

    @classmethod
    def smiles_features(cls, smiles, threshold = 40):
        graph_data = []
        for smile in smiles["smiles"]:
            try:
                if Chem.MolFromSmiles(smile).GetNumAtoms() <= threshold:
                    data = cls.smile_to_tensor(smile, threshold)
                    data.edge_feature = cls.smile_to_graph(smile)
                    graph_data.append(data)
            except:
                traceback.print_exc()
                pass
        return graph_data

    @classmethod
    def smiles_to_tensor(cls, smiles, threshold=40):
        """
        Use this preprocessing for data with no labels
        """
        BONDTYPE_TO_INT = defaultdict(
            lambda: 0,
            {
                BondType.ZERO: 0,
                BondType.SINGLE: 1,
                BondType.DOUBLE: 2,
                BondType.TRIPLE: 3,
                BondType.AROMATIC: 4
            }
        )
        graph_data = []
        for smile in smiles:
            try:
                if Chem.MolFromSmiles(smile).GetNumAtoms() <= threshold:
                    data = cls.smile_to_tensor(smile, threshold)
                    molecule = Chem.MolFromSmiles(smile)
                    n_atoms = molecule.GetNumAtoms()
                    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
                    n_edge_features = 5   
                    edge_features = np.zeros([40, 40, n_edge_features])
                    for bond in molecule.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
                        edge_features[i, j, bond_type] = 1
                        edge_features[j, i, bond_type] = 1
                    data.edge_ft = edge_features
                    graph_data.append(data)
            except:
                traceback.print_exc()
                pass
        return graph_data
    
    

    @classmethod
    def smile_to_graph(cls, smile):
        BONDTYPE_TO_INT = defaultdict(
            lambda: 0,
            {
                BondType.ZERO :0,
                BondType.SINGLE: 1,
                BondType.DOUBLE: 2,
                BondType.TRIPLE: 3,
                BondType.AROMATIC: 4
            }
        )
        
        molecule = Chem.MolFromSmiles(smile)
        n_atoms = molecule.GetNumAtoms()
        atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

        adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
        node_features = np.array([cls.atom_feature(atom) for atom in atoms])

        n_edge_features = 5
        edge_features = np.zeros([40, 40, n_edge_features])
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1
        edge_features = torch.tensor(edge_features)
        return edge_features


    @classmethod
    def smileToFPS(cls, path) -> FPS:
        data = pd.read_csv(path)
        smile = list(data['smiles'])
        for mol in smile:
            print(mol)
            mol = Chem.MolFromSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps = cls.fpsToArray(fp)
        return fps
        

    @classmethod
    def molToFps(cls, smiles, labels, fp_dim=2048):
        X = []
        Y = []
        for smile, label in zip(smiles, labels):
            X.append(smile)
            Y.append(label)
        return PrepDataset(X,Y)


    @classmethod
    def rxnToFps(cls, rxns, labels, fp_dim=2048):
        X = []
        Y = []
        for rxn, label in zip(rxns, labels):
            X.append(rxn)
            Y.append(label)
        return PrepDataset(X,Y)


    @classmethod
    def fingerprint_mols(cls, mols, fp_dim):
        for mol in [mols]:
            mol = Chem.MolFromSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim))
        return fp

    @classmethod
    def fingerprint_reactions(cls, reactions, fp_dim):
        for r in [reactions]:
            rxn = AllChem.ReactionFromSmarts(r)
            fp = AllChem.CreateStructuralFingerprintForReaction(rxn)
            fold_factor = fp.GetNumBits()//fp_dim
            fp = DataStructs.FoldFingerprint(fp, fold_factor)
        return fp

    @classmethod
    def fps_prep(cls, X):
        try:
            prod_mols, react_mols = X
            prod_fps = cls.fingerprint_mols(prod_mols, 8192)
            react_fps = cls.fingerprint_reactions(react_mols, 2048)
            return np.hstack([prod_fps, react_fps])
        except:
            traceback.print_exc()
            pass

    @classmethod
    def fps_preprocess(cls,data):

        X, y = data
        try:
            prod_mols, react_mols = X
            prod_fps = cls.fingerprint_mols(prod_mols, 8192)
            react_fps = cls.fingerprint_reactions(react_mols, 2048)
            return np.hstack([prod_fps, react_fps]), y
        except:
            traceback.print_exc()
            pass



    @classmethod
    def get_labels(cls,data):
        prod_to_rules = defaultdict(set)
        rxn_smiles, retro_temps = data
        for idx in tqdm(range(len(rxn_smiles))):
            prod_to_rules[rxn_smiles[idx].split('>')[2]].add(retro_temps[idx])
            
        print('prod_to_rules',len(prod_to_rules))


        print('In-Scope Filter training...')
        X, y = [], []
        exists = set()
        for prod, rules in tqdm(prod_to_rules.items(), desc='data prep'):
            rules = [r for r in rules]
            if not rules: continue

            for r in rules:
                if AllChem.ReactionFromSmarts(r):
                    if not isinstance(r, str):
                        print('False', r)
                    y.append(1.)
                    X.append((prod, r))
                    exists.add('{}_{}'.format(prod, r))

        print('X +ve',len(X))
        #Generate negative examples
        target_size = len(X) * 2
        pbar = tqdm(total=target_size//2, desc='data prep (negative)')
        prods = list(prod_to_rules.keys())
        exprules = list(prod_to_rules.values())
        while len(X) < target_size:
            prod = random.choice(prods)
            rule = random.choice(exprules)
            key = '{}_{}'.format(prod, r)
            try:
                if AllChem.ReactionFromSmarts(list(rule)[0]) and not Chem.MolFromSmiles(prod).HasSubstructMatch(Chem.MolFromSmarts(list(rule)[0].split('>')[0])):
                    if key in exists:
                        continue
                    else:
                        y.append(0.)
                        X.append((prod, list(rule)[0]))
                        pbar.update(1)
            except:
                continue
        pbar.close()
        return (X,y)


    @classmethod
    def fpsToArray(cls, fps) -> ND:
        arrs = []
        for fp in fps:
            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits())
            arr[onbits] = 1
            arrs.append(arr)
        arrs = np.array(arrs)
        return torch.from_numpy(arrs).float()


    @classmethod
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


    @classmethod
    def clean_data(cls, path, threshold=40):
        data = pd.read_csv(path)
        for index, row in data.iterrows():
            try:
                molecule = Chem.MolFromSmiles(data["smiles"][index])
                n_atoms = molecule.GetNumAtoms()
            except:
                data.drop(index, inplace = True)
        data["num_atoms"] = [Chem.MolFromSmiles(smile).GetNumAtoms() for smile in data['smiles']]
        data = data[data["num_atoms"] <= threshold]
        data = data.drop("num_atoms", axis = 1)
        smiles, labels = list(data["smiles"]), list(data['activity'])
        graph_data = cls.all_smile_to_tensor(smiles,labels)
        return graph_data


    @classmethod
    def find_clusters(cls,mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:  # special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append((a1, a2))

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        clusters.extend(ssr)

        atom_cls = [[] for i in range(n_atoms)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls


    @classmethod
    def extract_subgraph(cls,smiles, selected_atoms):
        # try with kekulization
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        subgraph, roots = __extract_subgraph(mol, selected_atoms)
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)

        mol = Chem.MolFromSmiles(smiles)  # de-kekulize
        if subgraph is not None and mol.HasSubstructMatch(subgraph):
            return Chem.MolToSmiles(subgraph), roots

        # If fails, try without kekulization
        subgraph, roots = __extract_subgraph(mol, selected_atoms)
        subgraph = Chem.MolToSmiles(subgraph)
        subgraph = Chem.MolFromSmiles(subgraph)
        if subgraph is not None:
            return Chem.MolToSmiles(subgraph), roots
        else:
            return None, None



    @classmethod
    def to_categorical(cls,y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


    @classmethod
    def preprocess_data(cls,dataset, human_vocab, machine_vocab, Tx, Ty):
        
        X, Y = zip(*dataset)
        
        X = np.array([cls.string_to_int(i, Tx, human_vocab) for i in X])
        Y = np.array([cls.string_to_int(t, Ty, machine_vocab) for t in Y])

        
        Xoh = np.array(list(map(lambda x: cls.to_categorical(x, num_classes=len(human_vocab)), X)))
        Yoh = np.array(list(map(lambda x: cls.to_categorical(x, num_classes=len(machine_vocab)), Y)))


        return X, Y, Xoh, Yoh

            
    @classmethod
    def string_to_int(cls,string, length, vocab):
        """
        Converts all strings in the vocabulary into a list of integers representing the positions of the
        input string's characters in the "vocab"
        
        Arguments:
        string -- input string
        length -- the number of time steps you'd like, determines if the output will be padded or cut
        vocab -- vocabulary, dictionary used to index every character of your "string"
        
        Returns:
        rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
        """

        u = vocab["<unk>"]   
        if len(string) > length:
            string = string[:length]
            
        rep = list(map(lambda x: vocab.get(x, u), string))
        
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(string))
        
        return rep


    @classmethod
    def int_to_string(cls,ints, inv_vocab):
        """
        Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
        
        Arguments:
        ints -- list of integers representing indexes in the machine's vocabulary
        inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
        
        Returns:
        l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
        """
        
        l = [inv_vocab[i] for i in ints]
        return l


    @classmethod
    def softmax(x, axis=-1):
        """Softmax activation function.
        # Arguments
            x : Tensor.
            axis: Integer, axis along which the softmax normalization is applied.
        # Returns
            Tensor, output of softmax transformation.
        # Raises
            ValueError: In case `dim(x) == 1`.
        """
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    

    @classmethod
    def canoSmiles(cls,smiles: str):
        try:
            tmp = Chem.MolFromSmiles(smiles)
            if tmp is None:
                return None, smiles        
            tmp = Chem.RemoveHs(tmp)
            if tmp is None:
                return None, smiles
            [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
            return tmp, Chem.MolToSmiles(tmp)            
        except:
            return None, smiles


    @classmethod
    def canoSmarts(cls,smarts: str):
        try:
            tmp = Chem.MolFromSmarts(smarts)
            # tmp.UpdatePropertyCache()   #Added by Babs
            if tmp is None:        
                return None, smarts
            [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
            cano = Chem.MolToSmarts(tmp)
            if '[[se]]' in cano:  # strange parse error
                cano = smarts
            return tmp, cano
        except:
            return None,smarts


    @classmethod
    def smarts_has_useless_parentheses(cls,smarts: str):
        if len(smarts) == 0:
            return False
        if smarts[0] != '(' or smarts[-1] != ')':
            return False
        cnt = 1
        for i in range(1, len(smarts)):
            if smarts[i] == '(':
                cnt += 1
            if smarts[i] == ')':
                cnt -= 1
            if cnt == 0:
                if i + 1 != len(smarts):
                    return False
        return True



    @classmethod
    def remove_space(cls,strs):
        sym_list = ['He',
        'Li',
        'Be',
        'Ne',
        'Na',
        'Mg',
        'Al',
        'Si',
        'Cl',
        'Ar',
        'Ca',
        'Sc',
        'Ti',
        'Cr',
        'Mn',
        'Fe',
        'Co',
        'Ni',
        'Cu',
        'Zn',
        'Ga',
        'Ge',
        'As',
        'Se',
        'Br',
        'Kr',
        'Rb',
        'Sr',
        'Zr',
        'Nb',
        'Mo',
        'Tc',
        'Ru',
        'Rh',
        'Pd',
        'Ag',
        'Cd',
        'In',
        'Sn',
        'Sb',
        'Te',
        'Xe',
        'Cs',
        'Ba',
        'La',
        'Ce',
        'Pr',
        'Nd',
        'Pm',
        'Sm',
        'Eu',
        'Gd',
        'Tb',
        'Dy',
        'Ho',
        'Er',
        'Tm',
        'Yb',
        'Lu',
        'Hf',
        'Ta',
        'Re',
        'Os',
        'Ir',
        'Pt',
        'Au',
        'Hg',
        'Tl',
        'Pb',
        'Bi',
        'Po',
        'At',
        'Rn',
        'Fr',
        'Ra',
        'Ac',
        'Th',
        'Pa',
        'Np',
        'Pu',
        'Am',
        'Cm',
        'Bk',
        'Cf',
        'Es',
        'Fm',
        'Md',
        'No',
        'Lr',
        'Rf',
        'Db',
        'Sg',
        'Bh',
        'Hs',
        'Mt',
        'Ds',
        'Rg',
        'Cn',
        'Nh',
        'Fl',
        'Mc',
        'Lv',
        'Ts',
        'Og']
        for sym in sym_list:
            try:
                s_char,e_char = list(sym)
                pat = re.search(rf"\b{s_char} {e_char}\b",strs).group(0)
                pat1= pat.replace(' ','') 
                strs = re.sub(pat,pat1, strs)
            except:
                continue
        return strs

    
    @classmethod
    def delete_space(cls,s):
        pat = re.compile(f'\s+(?=[^[\]]*\])')
        return re.sub(pat, "", s)


    @classmethod
    def to_categorical(cls,y: int, num_classes: int):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


    @classmethod
    def preprocess_data(cls,dataset: Any, human_vocab: dict, machine_vocab: dict, Tx, Ty):
        
        X,Y = zip(*dataset)
        
        X = np.array([cls.string_to_int(i, Tx, human_vocab) for i in X])
        Y = np.array([cls.string_to_int(t, Ty, machine_vocab) for t in Y])

        
        Xoh = np.array(list(map(lambda x: cls.to_categorical(x, num_classes=len(human_vocab)), X)))
        Yoh = np.array(list(map(lambda x: cls.to_categorical(x, num_classes=len(machine_vocab)), Y)))


        return X,Y,Xoh,Yoh
            

    @classmethod
    def string_to_int(cls, string, length, vocab):
        """
        Converts all strings in the vocabulary into a list of integers representing the positions of the
        input string's characters in the "vocab"
        
        Arguments:
        string -- input string
        length -- the number of time steps you'd like, determines if the output will be padded or cut
        vocab -- vocabulary, dictionary used to index every character of your "string"
        
        Returns:
        rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
        """

        u = vocab["<unk>"]   
        if len(string) > length:
            string = string[:length]
            
        rep = list(map(lambda x: vocab.get(x, u), string))
        
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(string))
        
        return rep


    @classmethod
    def int_to_string(cls, ints, inv_vocab):
        """
        Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
        
        Arguments:
        ints -- list of integers representing indexes in the machine's vocabulary
        inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
        
        Returns:
        l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
        """
        
        l = [inv_vocab[i] for i in ints]
        return l



    @classmethod
    def input_data(cls,data_path):
        dataset = []
        input_characters = set()
        target_characters = set()

        data_path = data_path
        df = pd.read_csv(data_path, nrows=5000)
        x = list(df['products'])
        y = list(df['reactants'])

    
        for input_text, target_text in zip(x,y):
            
            input_text = ' '.join(['<bos>',input_text,'<eos>'])
            target_text = ' '.join(['<bos>',target_text,'<eos>'])
            input_text = input_text.split(' ')
            target_text = target_text.split(' ')

            if len(input_text)<=50:

                ds = (input_text,target_text)
                dataset.append(ds)
                for char in input_text:
                    if char not in input_characters:
                        input_characters.add(char)
                for char in target_text:
                    if char not in target_characters:
                        target_characters.add(char)
                        
        z = np.array(dataset)
        print("length of dataset",len(z))

        Tx = 50
        Ty = 50

        input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
        target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']
        
        # target_characters.remove('')
        
        products_vocab = {v:k for k,v in enumerate(input_characters)}
        reactants_vocab = {v:k for k,v in enumerate(target_characters)}


        with open('../retrosynthesis/Seq2Seq/Transformer/reactants_vocab.json', 'w') as fr:
            json.dump(reactants_vocab, fr)
        with open('../retrosynthesis/Seq2Seq/Transformer/products_vocab.json', 'w') as fp:
            json.dump(products_vocab, fp)

        products_vocab_size = len(products_vocab)
        reactants_vocab_size = len(reactants_vocab)

        prod_pad = products_vocab['<pad>']
        react_pad = reactants_vocab["<pad>"]
        prod_bos = products_vocab['<bos>']
        react_bos = reactants_vocab['<bos>']

        inv_reactants_vocab = {v:k for k,v in reactants_vocab.items()} 

        X, Y, Xoh, Yoh = cls.preprocess_data(dataset, products_vocab, reactants_vocab, Tx, Ty)
        
        return RxnDataset(X, Y), products_vocab_size, reactants_vocab_size, prod_pad, react_pad

    @classmethod
    def fwd_input_data(cls,data_path):
        dataset = []
        input_characters = set()
        target_characters = set()

        data_path = data_path
        df = pd.read_csv(data_path, nrows=5000)
        
        x = list(df['reactants'])
        y = list(df['products'])

    
        for input_text, target_text in zip(x,y):
            
            input_text = ' '.join(['<bos>',input_text,'<eos>'])
            target_text = ' '.join(['<bos>',target_text,'<eos>'])
            input_text = input_text.split(' ')
            target_text = target_text.split(' ')

            if len(input_text)<=50:

                ds = (input_text,target_text)
                dataset.append(ds)
                for char in input_text:
                    if char not in input_characters:
                        input_characters.add(char)
                for char in target_text:
                    if char not in target_characters:
                        target_characters.add(char)
                        
        z = np.array(dataset)
        print("length of dataset",len(z))

        Tx = 50
        Ty = 50

        input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
        target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']

        
        # target_characters.remove('')
        
        reactants_vocab = {v:k for k,v in enumerate(input_characters)}
        products_vocab = {v:k for k,v in enumerate(target_characters)}


        with open('../retrosynthesis/Seq2Seq/Transformer/fwd_reactants_vocab.json', 'w') as fr:
            json.dump(reactants_vocab, fr)
        with open('../retrosynthesis/Seq2Seq/Transformer/fwd_products_vocab.json', 'w') as fp:
            json.dump(products_vocab, fp)


        reactants_vocab_size = len(reactants_vocab)
        products_vocab_size = len(products_vocab)

        prod_pad = products_vocab['<pad>']
        react_pad = reactants_vocab["<pad>"]
        prod_bos = products_vocab['<bos>']
        react_bos = reactants_vocab['<bos>']

        inv_products_vocab = {v:k for k,v in products_vocab.items()}

    

        inv_products_vocab = {v:k for k,v in products_vocab.items()} 

        X, Y, Xoh, Yoh = cls.preprocess_data(dataset, reactants_vocab, products_vocab, Tx, Ty)
        
        return RxnDataset(X, Y), products_vocab_size, reactants_vocab_size, prod_pad, react_pad


    @classmethod
    def get_graphfeatures(cls, sample: Any) -> Any:
        # convert the given smile to graph features [x, edge_index, label]
        tuplesmiles, y = sample
        smile1 = tuplesmiles[0]
        smile2 = tuplesmiles[1]
       
        mol = Chem.MolFromSmiles(smile1)
        mol1 = Chem.MolFromSmiles(smile2)
        edge_index, x = cls.graph_representation(mol, 40)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        
        edge_index1, x1 = cls.graph_representation(mol1, 40)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long)
        x1 = torch.tensor(x1, dtype=torch.float)
        
        y = torch.tensor(y, dtype=torch.float).view(1, -1)
        data = Data(x=x, edge_index=edge_index,x1=x1, edge_index1=edge_index1, y=y)
        return data

    """ it has to be merged with graph_features and made flexible"""
    @classmethod
    def get_graphfeatures1(cls, sample: Any) -> Any:
        # convert the given smile to graph features [x, edge_index, label]
        smile, y = sample
        mol = Chem.MolFromSmiles(smile)
        edge_index, x = PreprocessingFunc.graph_representation(mol, 40)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1, -1)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    @classmethod
    def getFwdLabels(cls,data):
        prod_to_rules = defaultdict(set)
        rxn_smiles = data['rxn_smiles']
        for idx in tqdm(range(len(rxn_smiles))):
            prod_to_rules[rxn_smiles[idx].split('>')[2]].add(rxn_smiles[idx].split('>')[0])
            
        print('prod_to_rules',len(prod_to_rules))


        print('In-Scope Filter training...')
        X, y = [], []
        exists = set()
        for prod, rules in tqdm(prod_to_rules.items(), desc='data prep'):
            rules = [r for r in rules]
            if not rules: continue

            for r in rules:
                if Chem.MolFromSmiles(r) and Chem.MolFromSmiles(prod):
                    if not isinstance(r, str):
                        print('False', r)
                    y.append(1.)
                    X.append((prod, r))
                    exists.add('{}_{}'.format(prod, r))

        print('X +ve',len(X))
        #Generate negative examples
        target_size = len(X) * 2
        pbar = tqdm(total=target_size//2, desc='data prep (negative)')
        prods = list(prod_to_rules.keys())
        exprules = list(prod_to_rules.values())
        while len(X) < target_size:
            prod = random.choice(prods)
            rule = random.choice(exprules)
            key = '{}_{}'.format(prod, r)
            if key in exists:
                continue
            else:
                y.append(0.)
                X.append((prod, list(rule)[0]))
                pbar.update(1)
        pbar.close()
        return (X,y)


