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

class MolNetPosDataset(InMemoryDataset):
    def __init__(self, root='/share/project/yuancheng/datasets/moleculenet/archived_with_h', smiles2graph=smiles2graph, transform=None, pre_transform=None, property=None): 
        '''
        '''
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.property = 'esol' if property is None else property
        self.folder = self.property
        super().__init__(osp.join(root, self.folder), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return self.property + '.pkl'


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    

    def mean(self, idxs):
        if self.property in ["freesolv", "esol", "lipo"]:
            y = torch.cat([self.get(i).y for i in idxs], dim=0)
            return y[:].mean().item()
        else:
            return 0.0

    def std(self, idxs):
        if self.property in ["freesolv", "esol", "lipo"]:
            y = torch.cat([self.get(i).y for i in idxs], dim=0)
            return y[:].std().item()
        else:
            return 1.0


    def process(self):
        with open(osp.join(self.root, self.raw_file_names), "rb") as f:
            data_df = pickle.load(f)
        smiles_list = data_df['smiles'].tolist()
        conf_list = data_df['conf'].tolist()
        conf_list_woh = [mol for mol in conf_list]
        property_list = data_df[data_df.columns[1:-1]]
        self.num_tasks = property_list.shape[1] 

        data_list = []
        suc, fail = 0, 0
        print('Preprocessing...')
        with Pool(processes=112) as pool:
            iter = pool.imap(mol2graph, conf_list_woh)

            for i, graph in tqdm(enumerate(iter), total=len(smiles_list)):
                try:
                    data = Data()
                    property = property_list.iloc[i]
                    property = ['nan' if x == '' else x for x in property]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])
                    assert (len(graph['node_feat'])) == graph['position'].shape[0]

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.tensor(np.array(property, dtype=np.float32))
                    data.pos = torch.from_numpy(graph['position']).to(torch.float32)

                    data_list.append(data)
                    suc += 1
                except:
                    fail += 1
        
    
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving processed data of ' + self.property)
        torch.save((data, slices), self.processed_paths[0])

        split = {}
        print(f"suc/fail: {suc}/{fail}")
        for seed in [0,1,2]:
            # scaffold split
            train, valid, test = scaffold_split(conf_list, balanced=True, seed=seed)            
            split['train'] = train
            split['valid'] = valid
            split['test'] = test
            print(f"Saving new split idx with seed{seed}")
            torch.save(split, osp.join(self.root, f"split_dict_seed{seed}.pt"))


    def get_idx_split(self, seed):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, f'split_dict_seed{seed}.pt')))
        for k, v in split_dict.items():
            split_dict[k] = torch.tensor(v)
        return split_dict
    


def mol2graph(mol):
    try:
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        # positions
        positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['position'] = positions

        return graph
    except:
        return None



def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        try:
            scaffold = generate_scaffold(mol)
        except:
            continue
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def scaffold_split(conf_list,
                   balanced: bool = True,
                   seed: int = 0,): 
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :return: A tuple containing the train, validation, and test split index.
    """

    # Split
    dataset_len = len(conf_list)
    train_size, val_size = 0.8 * dataset_len, 0.1 * dataset_len
    test_size = dataset_len - train_size - val_size
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(conf_list, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1


    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                    f'train scaffolds = {train_scaffold_count:,} | '
                    f'val scaffolds = {val_scaffold_count:,} | '
                    f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    # train = [data[i] for i in train]
    # val = [data[i] for i in val]
    # test = [data[i] for i in test]

    # return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    return train, val, test