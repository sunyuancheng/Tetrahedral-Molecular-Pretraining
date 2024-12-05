import os.path as osp
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data

from ogb.utils.torch_util import replace_numpy_with_torchtensor



class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    def __init__(self, root="/share/project/yuancheng/datasets/MD17_LEGO/", transform=None, pre_transform=None, dataset_arg=None, train_size=None, valid_size=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        self.molecule = dataset_arg
        self.root = root

        self.train_size = train_size
        self.valid_size = valid_size

        super(MD17, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, self.molecule + '_processed')

    @property
    def raw_file_names(self):
        return self.molecule + '_dft.npz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        assert len(self.raw_paths) == 1
        path = self.raw_paths[0]

        data_npz = np.load(path)
        z = torch.from_numpy(data_npz["z"]).long()
        positions = torch.from_numpy(data_npz["R"]).float()
        energies = torch.from_numpy(data_npz["E"]).float()
        forces = torch.from_numpy(data_npz["F"]).float()

        samples = []
        for pos, y, dy in zip(positions, energies, forces):
            samples.append(Data(x=z, pos=pos, y=y.unsqueeze(1), dy=dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

        # only support the 9500/500/remaining data split in Frad (Feng et al., 2023)
        # modify here for other data splits
        split = {}
        dset_len = len(samples)
        idxs = np.arange(dset_len, dtype=int)
        idxs = np.random.default_rng(seed=1).permutation(idxs)

        split['train'] = idxs[:self.train_size]
        split['valid'] = idxs[self.train_size : self.train_size + self.valid_size]
        split['test'] = idxs[self.train_size + self.valid_size : ]

        print(f"Saving new split idx with seed 1 and split {self.train_size}/{self.valid_size}/remaining")
        torch.save(split, osp.join(self.processed_dir, "split_dict.pt")) 

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.processed_dir, "split_dict.pt")))
        for k, v in split_dict.items():
            split_dict[k] = torch.tensor(v)
        return split_dict