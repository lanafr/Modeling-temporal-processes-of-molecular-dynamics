import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl

import sys, os

import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

from einops import rearrange

sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
print(sys.path)

NUM_OF_ATOMS = 258

class sequence_of_pos(Dataset):
    def __init__(self,
            dataset_path,
            device='cuda',
            split=(0.9, 0.1),
            seed_num=10,
            mode='train',
            length=20):

        self.device = device
        self.seed_num = seed_num
        self.dataset_path = dataset_path
        self.case_prefix = 'data_'
        self.length = length 
        self.how_many = 48

        self.mode = mode
        assert mode in ['train', 'test']
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        seed_seq = [i for i in range(seed_num*self.how_many)]
        ratio = split[0]
        if mode == 'train':
            self.index = seed_seq[:int(seed_num*ratio*self.how_many)]
        else:
            self.index = seed_seq[int(seed_num*ratio*self.how_many):]

    def __getitem__(self, index):

        data = []

        data.append(self.get_sequence(index))

        return data

    def __len__(self):
        return len(self.index)

    def get_sequence(self, index):
        seed = int(index//self.how_many)
        start = index%self.how_many*self.length
        idxs = np.arange(start, self.length+start)

        pos_lst = []
        vel_lst = []
        for_lst = []

        for i in idxs:

            current_pos = self.get_one(i, seed)['pos']
            pos_lst.append(current_pos)

            current_vel = self.get_one(i, seed)['vel']
            vel_lst.append(current_vel)

            current_for = self.get_one(i, seed)['forces']
            for_lst.append(current_for)

        dictionary = {'pos': pos_lst, 'vel': vel_lst, 'force': for_lst}

        return dictionary

    def get_one(self, idx, seed, get_path_name=False):
        
        fname = f'data_{seed}_{idx+1000}'

        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
            vel = raw_data['vel'].astype(np.float32)
            data['vel'] = vel

        if get_path_name:
            return data, data_path

        return data
    
class LJDataNew(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,   # per seed
                 case_prefix='data_',
                 seed_num=10,
                 split=(0.9, 0.1),
                 mode='train',
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.case_prefix = case_prefix
        self.seed_num = seed_num

        self.mode = mode
        assert mode in ['train', 'test']
        idxs = np.arange(seed_num*sample_num)
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        np.random.shuffle(idxs)
        ratio = split[0]
        if mode == 'train':
            self.idx = idxs[:int(len(idxs)*ratio)]
        else:
            self.idx = idxs[int(len(idxs)*ratio):]

    def __len__(self):
        return len(self.idx)


    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        seed = idx // self.sample_num
        fname = f'data_{seed}_{2*sample_to_read}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        #fname = f'data_3_678'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
            vel = raw_data['vel'].astype(np.float32)
            data['vel'] = vel
        if get_path_name:
            return data, data_path
        return data