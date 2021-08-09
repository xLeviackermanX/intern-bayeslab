from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

import random
import math
import time
import io
from os import sep

# from nltk.translate.bleu_score import sentence_bleu
import random
from sklearn.model_selection import train_test_split


from tqdm import tqdm
import sys
sys.path.append('../../')
import rdkit
from rdkit import Chem, DataStructs
import re
import traceback
from rdkit.Chem import AllChem
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type

import torchtext
from collections import Counter
from torchtext.vocab import Vocab

from torch.nn.utils.rnn import pad_sequence

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


random.seed(40)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]



def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])

    
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))


    return X, Y, Xoh, Yoh

def preprocess_single_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])

    
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))


    return X, Xoh, Y, Yoh



        
        
def string_to_int(string, length, vocab):
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
    
    #print (rep)
    return rep


def int_to_string(ints, inv_vocab):
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


class RxnDataset(torch.utils.data.Dataset):
    def __init__(self, list_inps,labels):
        self.labels = labels 
        self.list_inps = list_inps

    def __len__(self):
        return len(self.list_inps)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_inps[index]

        # Load data and get label
        X1 = ID
        #print('x',len(X),len(ID))
        y1 = self.labels[index]
        sample = (X1,y1)

        return sample



def input_data(data_path):
    dataset = []
    input_characters = set()
    target_characters = set()

    data_path = data_path
    df = pd.read_csv(data_path)
    x = list(df['products'])
    y = list(df['reactants'])

   
    for input_text, target_text in zip(x,y):
    # for input_text in lines[1:300]:
 
        # input_text, target_text = line.split('\t')
        input_text = ' '.join(['<bos>',input_text,'<eos>'])
        target_text = ' '.join(['<bos>',target_text,'<eos>'])
        input_text = input_text.split(' ')
        target_text = target_text.split(' ')

        # if len(input_text)>=len(target_text) and len(input_text)<=50:
        if len(input_text)<=50:

        # if len(input_text)<=20:
            ds = (input_text,target_text)
            # ds = input_text
            dataset.append(ds)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
                    
    z = np.array(dataset)
    print("length of dataset",len(z))

    # Tx = len(max(z[:,], key=len))
    # Ty = len(max(z[:,], key=len))
    Tx = 50
    Ty = 50

    input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
    target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']
    
    products_vocab = {v:k for k,v in enumerate(input_characters)}
    reactants_vocab = {v:k for k,v in enumerate(target_characters)}




    with open('/home/bayeslabs/molFlash/molflash/retrosynthesis/Seq2Seq/Transformer/reactants_vocab1.json', 'w') as fr:
        json.dump(reactants_vocab, fr)
    with open('/home/bayeslabs/molFlash/molflash/retrosynthesis/Seq2Seq/Transformer/products_vocab1.json', 'w') as fp:
        json.dump(products_vocab, fp)



    inv_reactants_vocab = {v:k for k,v in reactants_vocab.items()} 
    X, Y, Xoh, Yoh = preprocess_data(dataset, products_vocab, reactants_vocab, Tx, Ty)
    
    return RxnDataset(X,Y)


def input_single_data(data_path):
    dataset = []
    input_characters = set()
    target_characters = set()

    data_path = data_path
    df = pd.read_csv(data_path)
    x = list(df['products'])
    y = list(df['reactants'])

   
    for input_text, target_text in zip(x,y):
    # for input_text in lines[1:300]:
 
        # input_text, target_text = line.split('\t')
        input_text = ' '.join(['<bos>',input_text,'<eos>'])
        target_text = ' '.join(['<bos>',target_text,'<eos>'])
        input_text = input_text.split(' ')
        target_text = target_text.split(' ')

        # if len(input_text)>=len(target_text) and len(input_text)<=50:
        if len(input_text)<=50:

        # if len(input_text)<=20:
            ds = (input_text,target_text)
            # ds = input_text
            dataset.append(ds)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
                    
    z = np.array(dataset)
    print("length of dataset",len(z))

    # Tx = len(max(z[:,], key=len))
    # Ty = len(max(z[:,], key=len))
    Tx = 50
    Ty = 50

    input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
    target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']
    
    products_vocab = {v:k for k,v in enumerate(input_characters)}
    # print('reactants_vocab:',reactants_vocab)
    reactants_vocab = {v:k for k,v in enumerate(target_characters)}


   
    with open('/home/bayeslabs/molFlash/molflash/retrosynthesis/Seq2Seq/Transformer/products_vocab1.json', 'r') as f:
        product_vocab = json.load(f)

    with open('/home/bayeslabs/molFlash/molflash/retrosynthesis/Seq2Seq/Transformer/reactants_vocab1.json', 'r') as f:
        reactant_vocab = json.load(f)

    inv_reactants_vocab = {v:k for k,v in reactant_vocab.items()} 
    X,Xoh,Y,Yoh = preprocess_single_data(dataset, product_vocab, reactant_vocab, Tx, Ty)
 
    return X



def canoSmiles(smiles: str):
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


def canoSmarts(smarts: str):
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

def smarts_has_useless_parentheses(smarts: str):
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

def remove_space(strs):
    for sym in sym_list:
        try:
            s_char,e_char = list(sym)
            pat = re.search(rf"\b{s_char} {e_char}\b",strs).group(0)
            pat1= pat.replace(' ','') 
            strs = re.sub(pat,pat1, strs)
        except:
            continue
    return strs

def delete_space(s):
    pat = re.compile(f'\s+(?=[^[\]]*\])')
    return re.sub(pat, "", s)


def tokenize(string):
    string =  ' '.join([c for c in canoSmiles(string)[1]])
    string = remove_space(delete_space(string))
    return string


def build_vocab(filepath, tokenizer):
    counter = Counter()
    strings = list(pd.read_csv(filepath)['smiles'])
      
    for string_ in strings:
        counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])



def data_process(filepaths, source_vocab, target_vocab, tokenizer):
    raw_prod_iter = list(pd.read_csv(filepaths[0])['smiles'])
    raw_react_iter = list(pd.read_csv(filepaths[1])['smiles'])
    data = []
    for (raw_prod, raw_react) in zip(raw_prod_iter, raw_react_iter):
        prod_tensor_ = torch.tensor([source_vocab[token] for token in tokenizer(raw_prod)],
                            dtype=torch.long)
        react_tensor_ = torch.tensor([target_vocab[token] for token in tokenizer(raw_react)],
                            dtype=torch.long)
        data.append((prod_tensor_, react_tensor_))
    return data


def predict_preprocess(smiles, vocab, tokenizer):
    # smiles = pd.read_csv(filepath)['smiles']
    # smiles = list(smiles)
    # data = []
    # for raw_prod in smiles:
    prod_tensor_ = torch.tensor([vocab[token] for token in tokenizer(smiles)], dtype=torch.long)
    prod_batch=[torch.cat([torch.tensor([vocab['<bos>']]), prod_tensor_, torch.tensor([vocab['<eos>']])], dim=0)]
    prod_batch = pad_sequence(prod_batch, padding_value=vocab['<pad>'])
    # data.append(prod_batch)
    return prod_batch



def predict_preprocess(smiles, vocab, tokenizer):
    # smiles = pd.read_csv(filepath)['smiles']
    # smiles = list(smiles)
    # data = []
    # for raw_prod in smiles:
    prod_tensor_ = torch.tensor([vocab[token] for token in tokenizer(smiles)], dtype=torch.long)
    prod_batch=[torch.cat([torch.tensor([vocab['<bos>']]), prod_tensor_, torch.tensor([vocab['<eos>']])], dim=0)]
    prod_batch = pad_sequence(prod_batch, padding_value=vocab['<pad>'])
    # data.append(prod_batch)
    return prod_batch





