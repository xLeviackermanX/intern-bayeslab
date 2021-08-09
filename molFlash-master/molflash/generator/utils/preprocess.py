import io
from pandas.io.parsers import read_csv
from tqdm import tqdm
import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
import re
import traceback
from rdkit.Chem import AllChem
from rdkit import RDLogger

import torch
import torchtext
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import os

import logging
import tqdm


from sklearn.model_selection import train_test_split
import torch.utils.data as tud
from torch.autograd import Variable
import math

from molflash.models.transformerGEN.module.decode import decode
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
mpl.use('Agg')



def canoSmiles(smiles):
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


def canoSmarts(smarts):
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

def smarts_has_useless_parentheses(smarts):
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


def generator_preprocess(filepath, vocab, tokenizer):
    smiles = pd.read_csv(filepath)['smiles']
    smiles = list(smiles)


    data = []
    for raw_prod in smiles:
        prod_tensor_ = torch.tensor([vocab[token] for token in tokenizer(raw_prod)], dtype=torch.long)
        data.append(prod_tensor_)
    return data

def smile_preprocess(smiles, vocab, tokenizer):
    smiles = smiles
    prod_tensor_ = torch.tensor([vocab[token] for token in tokenizer(smiles)], dtype=torch.long)
    return prod_tensor_


def predict_preprocess(smiles, vocab, tokenizer):
    # smiles = pd.read_csv(filepath)['smiles']
    smiles = list(smiles)
    data = []
    for raw_prod in smiles:
        prod_tensor_ = torch.tensor([vocab[token] for token in tokenizer(raw_prod)], dtype=torch.long)
        prod_batch=[torch.cat([torch.tensor([vocab['<bos>']]), prod_tensor_, torch.tensor([vocab['<eos>']])], dim=0)]
        prod_batch = pad_sequence(prod_batch, padding_value=vocab['<pad>'])
        data.append(prod_batch)
    return data


def collate_fn(data_batch, source_vocab):
    prod_batch=[torch.cat([torch.tensor([source_vocab['<bos>']]), prod_item, torch.tensor([source_vocab['<eos>']])], dim=0) for prod_item in data_batch]
    prod_batch = pad_sequence(prod_batch, padding_value=source_vocab['<pad>'])
    return prod_batch




SEED = 42
SPLIT_RATIO = 0.8


def get_smiles_list(file_name):
    """
    Get smiles list for building vocabulary
    :param file_name:
    :return:
    """
    pd_data = pd.read_csv(file_name, sep=",")

    print("Read %s file" % file_name)
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    print("Number of SMILES in chemical transformations: %d" % len(smiles_list))

    return smiles_list

def split_data(input_transformations_path):
    """
    Split data into training, validation and test set, write to files
    :param input_transformations_path:L
    :return: dataframe
    """
    if type(input_transformations_path)==str:
        data = pd.read_csv(input_transformations_path, sep=",")

        train, test = train_test_split(data, test_size=0.1, random_state=SEED)
        train, validation = train_test_split(train, test_size=0.1, random_state=SEED)
        return train, validation, test
    
    train, test = train_test_split(input_transformations_path, test_size=0.1, random_state=SEED)
    # parent = get_parent_dir(input_transformations_path)
    # train.to_csv(os.path.join(parent, "train.csv"), index=False)
    # validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
    # test.to_csv(os.path.join(parent, "test.csv"), index=False)

    return train, validation, test

def save_df_property_encoded(file_name, property_change_encoder):
    data = pd.read_csv(file_name, sep=",")
    PROPERTIES = ['LogD', 'Solubility', 'Clint']
    PROPERTY_THRESHOLD = {
    'Solubility': math.log(50, 10),
    'Clint': math.log(20, 10)
}
    for property_name in PROPERTIES:
        if property_name == 'LogD':
            encoder, start_map_interval = property_change_encoder[property_name]
            data['Delta_{}'.format(property_name)] = \
                data['Delta_{}'.format(property_name)].apply(lambda x:
                                                                value_in_interval(x, start_map_interval), encoder)
        elif property_name in ['Solubility', 'Clint']:
            data['Delta_{}'.format(property_name)] = data.apply(
                lambda row: prop_change(row['Source_Mol_{}'.format(property_name)],
                                        row['Target_Mol_{}'.format(property_name)],
                                        PROPERTY_THRESHOLD[property_name]), axis=1)

    # output_file = file_name.split('.csv')[0] + '_encoded.csv'
    # data.to_csv(output_file, index=False)
    # return output_file
    return data

def prop_change(source, target, threshold):
    if source <= threshold and target > threshold:
        return "low->high"
    elif source > threshold and target <= threshold:
        return "high->low"
    elif source <= threshold and target <= threshold:
        return "no_change"
    elif source > threshold and target > threshold:
        return "no_change"





STEP = 0.2

def encode_property_change(input_data_path):
    property_change_encoder = {}
    PROPERTIES = ['LogD', 'Solubility', 'Clint']
    for property_name in PROPERTIES:
        if property_name == 'LogD':
            intervals, start_map_interval = build_intervals(input_data_path, step=STEP)
        elif property_name in ['Solubility', 'Clint']:
            intervals = ['low->high', 'high->low', 'no_change']

        if property_name == 'LogD':
            property_change_encoder[property_name] = intervals, start_map_interval
        elif property_name in ['Solubility', 'Clint']:
            property_change_encoder[property_name] = intervals

    return property_change_encoder


def value_in_interval(value, start_map_interval):
    start_vals = sorted(list(start_map_interval.keys()))
    return start_map_interval[start_vals[np.searchsorted(start_vals, value, side='right') - 1]]


def interval_to_onehot(interval, encoder):
    return encoder.transform([interval]).toarray()[0]


def build_intervals(input_transformations_path, step=STEP):
    df = pd.read_csv(input_transformations_path)
    delta_logD = df['Delta_LogD'].tolist()
    min_val, max_val = min(delta_logD), max(delta_logD)


    start_map_interval = {}
    interval_str = '({}, {}]'.format(round(-step/2, 2), round(step/2, 2))
    intervals = [interval_str]
    start_map_interval[-step/2] = interval_str

    positives = step/2
    while positives < max_val:
        interval_str = '({}, {}]'.format(round(positives, 2), round(positives+step, 2))
        intervals.append(interval_str)
        start_map_interval[positives] = interval_str
        positives += step
    interval_str = '({}, inf]'.format(round(positives, 2))
    intervals.append(interval_str)
    start_map_interval[float('inf')] = interval_str

    negatives = -step/2
    while negatives > min_val:
        interval_str = '({}, {}]'.format(round(negatives-step, 2), round(negatives, 2))
        intervals.append(interval_str)
        negatives -= step
        start_map_interval[negatives] = interval_str
    interval_str = '(-inf, {}]'.format(round(negatives, 2))
    intervals.append(interval_str)
    start_map_interval[float('-inf')] = interval_str

    return intervals, start_map_interval




class Vocabulary:
    """Stores the tokens and their conversion to one-hot vectors."""

    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            raise ValueError("Token already present in the vocabulary")
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens, encoding them in 1-hot encoded vectors."""
        ohe_vect = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            ohe_vect[i] = self._tokens[token]
        return ohe_vect

    def decode(self, ohe_vect):
        """Decodes a one-hot encoded vector matrix to a list of tokens."""
        tokens = []
        for ohv in ohe_vect:
            tokens.append(self[ohv])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]

    def word2idx(self):
        return {k: self._tokens[k] for k in self._tokens if isinstance(k, str)}





class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer, property_condition=None):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["*", "^", "$"] + sorted(tokens))  # pad=0, start=1, end=2
    if property_condition is not None:
        vocabulary.update(property_condition)
    # for random smiles
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])

    return vocabulary




def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0




class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing
    Source_Mol_ID,Target_Mol_ID,Source_Mol,Target_Mol,
    Source_Mol_LogD,Target_Mol_LogD,Delta_LogD,
    Source_Mol_Solubility,Target_Mol_Solubility,Delta_Solubility,
    Source_Mol_Clint,Target_Mol_Clint,Delta_Clint,
    Transformation,Core"""

    def __init__(self, data, vocabulary, tokenizer, prediction_mode=False):
        """
        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        :param prediction_mode: if use target smiles or not (training or test)
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data
        self._prediction_mode = prediction_mode

    def __getitem__(self, i):
        """
        Tokenize and encode source smile and/or target smile (if prediction_mode is True)
        :param i:
        :return:
        """

        row = self._data.iloc[i]

        # tokenize and encode source smiles
        source_smi = row['Source_Mol']
        source_tokens = []
        # source_change = []
        PROPERTIES = ['LogD', 'Solubility', 'Clint']
        for property_name in PROPERTIES:
            if property_name == 'LogD':
                source_tokens.append(row['Delta_{}'.format(property_name)])
            else:
                change = row['Delta_{}'.format(property_name)]
                source_tokens.append(f"{property_name}_{change}")

        source_tokens.extend(self._tokenizer.tokenize(source_smi))
        source_encoded = self._vocabulary.encode(source_tokens)

        # tokenize and encode target smiles if it is for training instead of evaluation
        if not self._prediction_mode:
            target_smi = row['Target_Mol']
            target_tokens = self._tokenizer.tokenize(target_smi)
            target_encoded = self._vocabulary.encode(target_tokens)

            return torch.tensor(source_encoded, dtype=torch.long), torch.tensor(
                target_encoded, dtype=torch.long), row
        else:
            return torch.tensor(source_encoded, dtype=torch.long),  row

    def __len__(self):
        return len(self._data)

    @classmethod
    def collate_fn(cls, data_all):
        # sort based on source sequence's length
        data_all.sort(key=lambda x: len(x[0]), reverse=True)
        is_prediction_mode = True if len(data_all[0]) == 2 else False
        if is_prediction_mode:
            source_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)
        else:
            source_encoded, target_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)

        # maximum length of source sequences
        max_length_source = max([seq.size(0) for seq in source_encoded])
        # padded source sequences with zeroes
        collated_arr_source = torch.zeros(len(source_encoded), max_length_source, dtype=torch.long)
        for i, seq in enumerate(source_encoded):
            collated_arr_source[i, :seq.size(0)] = seq
        # length of each source sequence
        source_length = [seq.size(0) for seq in source_encoded]
        source_length = torch.tensor(source_length)
        # mask of source seqs
        src_mask = (collated_arr_source !=0).unsqueeze(-2)

        # target seq
        if not is_prediction_mode:
            max_length_target = max([seq.size(0) for seq in target_encoded])
            collated_arr_target = torch.zeros(len(target_encoded), max_length_target, dtype=torch.long)
            for i, seq in enumerate(target_encoded):
                collated_arr_target[i, :seq.size(0)] = seq

            trg_mask = (collated_arr_target != 0).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask))
            trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token
        else:
            trg_mask = None
            max_length_target = None
            collated_arr_target = None

        return collated_arr_source, source_length, collated_arr_target, src_mask, trg_mask, max_length_target, data




def get_canonical_smile(smile):
    if smile != 'None':
        mol = rkc.MolFromSmiles(smile)
        if mol is not None:
            smi = rkc.MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
            return smi
        else:
            return None
    else:
        return None


def is_valid(smi):
    return 1 if to_mol(smi) else 0

def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return rkc.MolFromSmiles(smi)



def sample(model, src, src_mask, source_length, decode_type, vocab, tokenizer = SMILESTokenizer(), num_samples=10,max_len=75):
    batch_size = src.shape[0]
    num_valid_batch = np.zeros(batch_size)  # current number of unique and valid samples out of total sampled
    num_valid_batch_total = np.zeros(batch_size)  # current number of sampling times no matter unique or valid
    num_valid_batch_desired = np.asarray([num_samples] * batch_size)
    unique_set_num_samples = [set()] * batch_size  # for each starting molecule
    batch_index = torch.LongTensor(range(batch_size))
    batch_index_current = torch.LongTensor(range(batch_size))
    start_mols = []
    # zeros correspondes to ****** which is valid according to RDKit
    sequences_all = torch.ones((num_samples, batch_size, max_len))
    sequences_all = sequences_all.type(torch.LongTensor)
    max_trials = 100  # Maximum trials for sampling
    current_trials = 0

    if decode_type == 'greedy':
        max_trials = 1

    # Set of unique starting molecules
    if src is not None:
        for ibatch in range(batch_size):
            source_smi = tokenizer.untokenize(vocab.decode(src[ibatch].tolist()[3:]))
            source_smi = get_canonical_smile(source_smi)
            unique_set_num_samples[ibatch].add(source_smi)
            start_mols.append(source_smi)

    with torch.no_grad():
        while not all(num_valid_batch >= num_valid_batch_desired) and current_trials < max_trials:
            current_trials += 1
            if src is not None:
                    src_current = src.index_select(0, batch_index_current)
            if src_mask is not None:
                mask_current = src_mask.index_select(0, batch_index_current)
            batch_size = src_current.shape[0]

            sequences = decode(model, src_current, mask_current, max_len, decode_type)
            padding = (0, max_len-sequences.shape[1],
                        0, 0)
            sequences = torch.nn.functional.pad(sequences, padding)
            

            # Check valid and unique
            smiles = []
            is_valid_index = []
            batch_index_map = dict(zip(list(range(batch_size)), batch_index_current))
            # Valid, ibatch index is different from original, need map back
            for ibatch in range(batch_size):
                seq = sequences[ibatch]
                smi = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                smi = get_canonical_smile(smi)
                smiles.append(smi)
                # valid and not same as starting molecules
                if is_valid(smi):
                    is_valid_index.append(ibatch)
                # total sampled times
                num_valid_batch_total[batch_index_map[ibatch]] += 1

            # Check if duplicated and update num_valid_batch and unique
            for good_index in is_valid_index:
                index_in_original_batch = batch_index_map[good_index]
                if smiles[good_index] not in unique_set_num_samples[index_in_original_batch]:
                    unique_set_num_samples[index_in_original_batch].add(smiles[good_index])
                    num_valid_batch[index_in_original_batch] += 1

                    sequences_all[int(num_valid_batch[index_in_original_batch] - 1), index_in_original_batch, :] = \
                        sequences[good_index]

            not_completed_index = np.where(num_valid_batch < num_valid_batch_desired)[0]
            if len(not_completed_index) > 0:
                batch_index_current = batch_index.index_select(0, torch.LongTensor(not_completed_index))

    # Convert to SMILES
    smiles_list = [] # [batch, topk]
    seqs = np.asarray(sequences_all.numpy())
    # [num_sample, batch_size, max_len]
    batch_size = len(seqs[0])
    for ibatch in range(batch_size):
        topk_list = []
        for k in range(num_samples):
            seq = seqs[k, ibatch, :]
            topk_list.extend([tokenizer.untokenize(vocab.decode(seq))])
        smiles_list.append(topk_list)


    return smiles_list



"""
RDKit util functions.
"""
import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit import DataStructs

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()

def to_fp_ECFP(smi):
    if smi:
        mol = rkc.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprint(mol, 2)

def tanimoto_similarity_pool(args):
    return tanimoto_similarity(*args)

def tanimoto_similarity(smi1, smi2):
    fp1, fp2 = None, None
    if smi1 and type(smi1)==str and len(smi1)>0:
        fp1 = to_fp_ECFP(smi1)
    if smi2 and type(smi2)==str and len(smi2)>0:
        fp2 = to_fp_ECFP(smi2)

    if fp1 is not None and fp2 is not None:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return None

def is_valid(smi):
    return 1 if to_mol(smi) else 0

def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return rkc.MolFromSmiles(smi)

def get_canonical_smile(smile):
    if smile != 'None':
        mol = rkc.MolFromSmiles(smile)
        if mol is not None:
            smi = rkc.MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
            return smi
        else:
            return None
    else:
        return None



def make_directory(file, is_dir=True):
    dirs = file.split('/')[:-1] if not is_dir else file.split('/')
    path = ''
    for dir in dirs:
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.makedirs(path)

def get_parent_dir(file):
    dirs = file.split('/')[:-1]
    path = ''
    for dir in dirs:
        path = os.path.join(path, dir)
    if file.startswith('/'):
        path = '/' + path
    return path

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out




def get_logger(name, log_path, level=logging.INFO):
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Logging to a file
    make_directory(log_path, is_dir=False)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def progress_bar(iterable, total, **kwargs):
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)


global LOG
LOG = get_logger("preprocess", "experiments/preprocess.log")


def hist_box(data_frame, field, name="hist_box", path="./", title=None):

    title = title if title else field
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    data_frame[field].plot.hist(bins=100, title=title, ax=axs[0])
    data_frame.boxplot(field, ax=axs[1])
    plt.title(title)
    plt.suptitle("")

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def hist(data_frame, field, name="hist", path="./", title=None):


    title = title if title else field

    plt.hist(data_frame[field])
    plt.title(title)
    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def hist_box_list(data_list, name="hist_box", path="./", title=None):

    fig, axs = plt.subplots(1,2,figsize=(10,4))
    axs[0].hist(data_list, bins=100)
    axs[0].set_title(title)
    axs[1].boxplot(data_list)
    axs[1].set_title(title)

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def scatter_hist(x, y, name, path, field=None, lims=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    n = len(x)
    xy = np.vstack([x+ 0.00001 * np.random.rand(n), y+ 0.00001 * np.random.rand(n)])
    z = gaussian_kde(xy)(xy)
    axs[0].scatter(x, y, c=z, s=3, marker='o', alpha=0.2)
    lims = [np.min([axs[0].get_xlim(), axs[0].get_ylim()]), np.max([axs[0].get_xlim(), axs[0].get_ylim()])] if lims is None else lims
    axs[0].plot(lims, lims, 'k-', alpha=0.75)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(lims)
    axs[0].set_ylim(lims)
    xlabel = ""
    ylabel = ""
    if "delta" in field:
        if "data" in field:
            axs[0].set_xlabel(r'$\Delta LogD$ (experimental)')
            axs[0].set_ylabel(r'$\Delta LogD$ (calculated)')
            xlabel = 'Delta LogD (experimental)'
            ylabel = 'Delta LogD (calculated)'
        elif "predict" in field:
            axs[0].set_xlabel(r'$\Delta LogD$ (desirable)')
            axs[0].set_ylabel(r'$\Delta LogD$ (generated)')
            xlabel = 'Delta LogD (desirable)'
            ylabel = 'Delta LogD (generated)'
    if "single" in field:
        if "data" in field:
            xlabel, ylabel = 'LogD (experimental)', 'LogD (calculated)'
            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel)
        elif "predict" in field:
            xlabel, ylabel = 'LogD (desirable)', 'LogD (generated)'
            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel)
    bins = np.histogram(np.hstack((x, y)), bins=100)[1]  # get the bin edges
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=bins, stacked=False)
    axs[1].hist(x, **kwargs, color='b', label=xlabel)
    axs[1].hist(y, **kwargs, color='r', label=ylabel)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()







