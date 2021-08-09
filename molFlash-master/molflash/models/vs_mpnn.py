import rdkit
import rdkit.Chem as Chem
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils import *
# from data_preprocess import *

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

ELEM_LIST = ['C', 'N', 'O', 'S','F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
### basic setting from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/mpn.py

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor( onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                   'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']


def smiles2mpnnfeature(smiles):
    ## mpn.py::tensorize
    '''
        data-flow:
            data_process(): apply(smiles2mpnnfeature)
            DBTA: train(): data.DataLoader(data_process_loader())
            mpnn_collate_func()
    '''
    try:
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms, fbonds = [], [padding]
        in_bonds,all_bonds = [], [(-1,-1)]
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append( atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx()
            y = a2.GetIdx()

            b = len(all_bonds)
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(n_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
        for a in range(n_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2
    except:
        print('Molecules not found and change to zero vectors..')
        fatoms = torch.zeros(0,39)
        fbonds = torch.zeros(0,50)
        agraph = torch.zeros(0,6)
        bgraph = torch.zeros(0,6)
    #fatoms, fbonds, agraph, bgraph = [], [], [], []
    #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
    Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
    shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
    #print(shape_tensor)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]



def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

class MPNN(nn.Module):
    def __init__(self, depth,out_dim):
        super(MPNN, self).__init__()
        self.mpnn_hidden_size = out_dim
        self.mpnn_depth = depth
#         from DeepPurpose.chemutils import ATOM_FDIM, BOND_FDIM

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)


    ### first version, forward single molecule sequentially.
    def forward(self, feature):
        '''
            batch_size == 1
            feature: utils.smiles2mpnnfeature
        '''
        fatoms, fbonds, agraph, bgraph, atoms_bonds= feature
        agraph = agraph.long()
        bgraph = bgraph.long()
        #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape, atoms_bonds.shape)
        atoms_bonds = atoms_bonds.long()
        batch_size = atoms_bonds.shape[0]
        N_atoms, N_bonds = 0, 0
        embeddings = []
        for i in range(batch_size):
            n_a = atoms_bonds[i,0].item()
            n_b = atoms_bonds[i,1].item()
            if (n_a == 0):
                embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
                embeddings.append(embed)
                continue
            sub_fatoms = fatoms[N_atoms:N_atoms+n_a,:]
            sub_fbonds = fbonds[N_bonds:N_bonds+n_b,:]
            sub_agraph = agraph[N_atoms:N_atoms+n_a,:]
            sub_bgraph = bgraph[N_bonds:N_bonds+n_b,:]
            embed = self.single_molecule_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
            embed = embed
            embeddings.append(embed)
            N_atoms += n_a
            N_bonds += n_b
        try:
            embeddings = torch.cat(embeddings, 0)
        except:
            #embeddings = torch.cat(embeddings, 0)
            print(embeddings)
        return embeddings



    def single_molecule_forward(self, fatoms, fbonds, agraph, bgraph):
        '''
            fatoms: (x, 39)
            fbonds: (y, 50)
            agraph: (x, 6)
            bgraph: (y,6)
        '''
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        message = F.relu(binput)
        #print("shapes", fbonds.shape, binput.shape, message.shape)
        for i in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))
        return torch.mean(atom_hiddens, 0).view(1,-1)

#
if __name__=="__main__":
    print("praven")
    # mpnn_cnn_config = { 'depth': 3,
    #                     'drug_output': 256,
    #                     'input_dim_protein': 26,  # input of base protein encoding (26,1000)
    #                     'output_dim_protein': 256,  # output of the protein encoding(desired in this case 256)
    #                     'cnn_target_filters': [32, 64, 96],  # filters for the  CNN
    #                     'cnn_target_kernels': [4, 8, 12],  # kernals for CNN
    #                     'batch_size': 10,  ##########
    #                     'LR': 0.003,  ####################
    #                     'num_workers': 4}

    smile = 'n1ccccc1'
    features = smiles2mpnnfeature(smile)
    print(features[0].shape)

    en = MPNN(3,256)  # no of dimension and depth are the parameters
    print(f"finnnnnnnnnn",en(features).shape)

