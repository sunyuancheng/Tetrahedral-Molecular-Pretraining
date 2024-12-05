# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
import torch
import numpy as np
import os.path as osp
import pickle
import tqdm

from ogb.utils.torch_util import replace_numpy_with_torchtensor
from multiprocessing import Pool
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from typing import List, Union, Dict, Set
import random
from ogb.utils import smiles2graph

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

class LBADataset(InMemoryDataset):
    def __init__(self, root='/share/project/yuancheng/datasets/LBA_atom3d/FradNMI_datadict', subset="split_30", split='train'): 
        '''
        Use pre-processed npy data files as in Frad.
        https://github.com/fengshikun/FradNMI/blob/main/torchmdnet/datasets/atom3dlba.py
        '''
        assert subset=="split_30" or subset=="split_60"
        assert split in ['train', 'valid', 'test']
        
        self.original_root = root
        self.subset = subset
        self.split = split

        self.data_dict = np.load(osp.join(root, self.subset, f'lba_{self.split}.npy'), allow_pickle=True).item()

        self.length = len(self.data_dict['index'])

        # TODO: the hard code, to  adapt to collator_3d
        if subset == "split_30":
            self.train_mean = 6.5239
            self.train_std = 2.0012
        elif subset == "split_60":
            self.train_mean = 6.4793
            self.train_std = 1.9701
        else:
            raise NotImplementedError


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        
        data = Data()

        num_atoms = self.data_dict['num_atoms'][idx]
        pocket_atomsnum = self.data_dict['pocket_atoms'][idx]
        ligand_atomsnum = self.data_dict['ligand_atoms'][idx]
        assert (pocket_atomsnum + ligand_atomsnum) == num_atoms
        data.x = torch.tensor(self.data_dict['charges'][idx][:num_atoms], dtype=torch.long)
        data.pos = torch.tensor(self.data_dict['positions'][idx][:num_atoms], dtype=torch.float32)
        
        data.y =  torch.tensor(self.data_dict['neglog_aff'][idx], dtype=torch.float32)

        data.pocket_atomsnum = pocket_atomsnum
        
        # type mask
        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[pocket_atomsnum: ] = 1 # lig 1
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        return data
