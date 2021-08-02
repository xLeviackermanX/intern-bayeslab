

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import time
import random
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
import rdkit
from rdkit import Chem, DataStructs
import re
import traceback
from rdkit.Chem import AllChem
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Type

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


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


def prepareData(path='~/proc_train_singleprod.csv'):
    rxn_smiles = pd.read_csv(path)
    rxns = rxn_smiles['rxn_smiles']
    # reactants = []
    products = []
    for rxn in tqdm(rxns):
        reacts,_,prod = rxn.split('>')
        # cano_reacts = ' '.join([c for c in canoSmiles(reacts)[1]])
        cano_prod = ' '.join([c for c in canoSmiles(prod)[1]])

        # reactants.append(remove_space(delete_space(cano_reacts)))
        products.append(remove_space(delete_space(cano_prod)))
    # df = pd.DataFrame({'reactants':reactants,'products':products})
    df = pd.DataFrame({'products':products})


    df.to_csv('test_data.csv', index=False,sep='\t',header=None)






def to_categorical(y: int, num_classes: int):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]




random.seed(40)

def preprocess_data(dataset: Any, human_vocab: dict, Tx):
    
    X = dataset
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    # X = torch.from_numpy(X)
    # Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])
    # print("X", X.shape)
    # print("Y", Y.shape)

    
    # Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    # Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    # print("Xoh", Xoh.shape)
    # print("Yoh", Yoh.shape)

    # print(Xoh[0])

    return torch.from_numpy(X).long()
        
        
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




def input_data(data_path):
    '''
    Args:
        1.data_path
    '''
    dataset = []
    input_characters = set()
    # target_characters = set()

    data_path = data_path
    df = pd.read_csv(data_path,nrows=100)
    lines = list(df['product'])
    print(len(lines))

    
    for input_text in lines:
 
        # input_text, target_text = line.split('\t')
        input_text = ' '.join(['<SRXN>',input_text,'<ERXN>'])
        # target_text = ' '.join(['<SRXN>',target_text,'<ERXN>'])
        input_text = input_text.split(' ')
        # target_text = target_text.split(' ')

        # if len(input_text)>=len(target_text) and len(input_text)<=30:
        if len(input_text)<=50:
            # ds = (input_text,target_text)
            ds = input_text
            dataset.append(ds)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            # for char in target_text:
            #     if char not in target_characters:
            #         target_characters.add(char)
                    
    z = np.array(dataset)
    print(len(z))

    Tx = len(max(z[:,], key=len))
    # Tx = 30
    # Ty = len(max(z[:,1], key=len))

    input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']
    # target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']
    
    input_vocab = {v:k for k,v in enumerate(input_characters)}
    # print('reactants_vocab:',reactants_vocab)
    # products_vocab = {v:k for k,v in enumerate(target_characters)}


    with open('/home/bayeslabs/molFlash/molflash/generator/Seq2Seq/input_vocab.json', 'w') as fr:
        json.dump(input_vocab, fr)
    X = preprocess_data(dataset, input_vocab, Tx)

    return X, input_vocab


