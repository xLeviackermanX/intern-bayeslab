import torch
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
from MPNN import *
from torch.utils.data.dataloader import default_collate
from models.CNN import *
import pandas as pd
from utils.preprocess import *


from data import PreprocessingFunc
from models.vs_cnn import CNN
from models.vs_mpnn import MPNN


class CustomTask(flash.Task):
    def __init__(self, path, batchsize=10):
        super(viruvalscreening, self).__init__()
        self.batch_size = batchsize
        self.mpnn = MPNN(256, 3)
        self.cnn = CNN('protein')
        self.path = path

        """some changes to be done"""
        self.layer_1 = nn.Linear(512, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 1)
        self.layer_5 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)

    def forward(self, x, y):
        """
        Call the MPNN and CNN Network
        Combine the output and
        apply the additional network layers

        :param x: input for mpnn
        :param y: input for cnn
        :return:
        """
        d = self.mpnn(x)
        p = self.cnn(y)
        v_f = torch.cat([d, p], 1)
        l1 = self.layer_1(v_f)
        d1 = self.drop1(l1)
        l2 = self.layer_2(d1)
        d2 = self.drop2(l2)
        l3 = self.layer_3(d2)
        d3 = self.drop3(l3)
        l4 = self.layer_4(d3)
        l5 = self.layer_5(l4)
        return l5

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y, z = batch

        y_hat = self.forward(x, y)
        mse = torch.nn.MSELoss()
        loss = mse(y_hat,z.unsqueeze(1))
        # logs = {'loss': loss}
        return {'loss': loss}
    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat = self.forward(x, y)
        mse = torch.nn.MSELoss()
        loss = mse(y_hat, z.unsqueeze(1))
        # logs = {'loss': loss}
        return {'loss': loss}

class CustomPreprocess(Preprocess):
    def mpnn_feature_collate_func(x):
        N_atoms_scope = torch.cat([i[4] for i in x], 0)
        f_a, f_b, agraph_lst, bgraph_lst = [], [], [], []

        for j in range(len(x)):
            f_a.append(x[j][0])
            f_b.append(x[j][1])
            agraph_lst.append(x[j][2])
            bgraph_lst.append(x[j][3])
        agraph = torch.cat(agraph_lst, 0)
        bgraph = torch.cat(bgraph_lst, 0)
        f_a = torch.cat(f_a, 0)
        f_b = torch.cat(f_b, 0)
        return [f_a, f_b, agraph, bgraph, N_atoms_scope]

    ## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
    def mpnn_collate_func(x):
        """MPNN network collate function"""
        mpnn_feature = [i[0] for i in x]
        mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
        from torch.utils.data.dataloader import default_collate
        x_remain = [list(i[1:]) for i in x]
        x_remain_collated = default_collate(x_remain)
        return [mpnn_feature] + x_remain_collated





class customDataModule(DataModule):

    def get_data(self,configargs):
        """read a csv file"""
        return(pd.read_csv(file,configargs.file))


    def prepare_data(self):
        """
        data split and call dataloaders
        :return:
        """
        train, val, test = preprocess_data(self.path, frac=[0.8, 0.2, 0])
        # PROBLEM : dont use the same variable names heres bcz train above is dataframe and if you use train1 place as train it gives error bcz dataprocesloader is continues runnig
        self.train1 = data_process_loader(train)
        self.val1 = data_process_loader(val)
        self.test1 = data_process_loader(test)


    def train_dataloader(self):
        return DataLoader(self.train1, self.batch_size, collate_fn=mpnn_collate_func, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val1, self.batch_size, collate_fn=mpnn_collate_func, num_workers=4)





if __name__='__main__':
        kk=0#config file loading
