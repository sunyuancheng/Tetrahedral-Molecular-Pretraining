# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset, GraphormerPYGDatasetQM9
from .molnet import MolNetPosDataset
from .qm9 import newQM9, newHQM9
from .md17 import MD17
from .lba import LBADataset
import torch.distributed as dist


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyHQM9(newHQM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNetPos(MolNetPosDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNetPos, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNetPos, self).process()
        if dist.is_initialized():
            dist.barrier()



class MyMD17(MD17):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMD17, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMD17, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyLBA(LBADataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyLBA, self).download()
        if dist.is_initialized():
            dist.barrier()

def process(self):
    if not dist.is_initialized() or dist.get_rank() == 0:
        super(MyLBA, self).process()
    if dist.is_initialized():
        dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int, dataset_path: str, train_size: int, valid_size: int, task_idx: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        qm9_data = False
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
            qm9_data = True
        elif name == "qm9H":
            inner_dataset = MyHQM9(root=root)
            qm9_data = True
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name.startswith("md17"):
            md17_molecule = name.split("-")[1]
            assert (train_size==9500 and valid_size==500), "For other data split sizes, modify the dataset self.get_idx_split and create data split in advance"

            # data root hardcoded in dataset class
            inner_dataset = MyMD17(train_size=train_size, valid_size=valid_size, dataset_arg=md17_molecule)
            print(f"load {md17_molecule} dataset")

            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
            return GraphormerPYGDataset(
                inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
            
        elif name.startswith("molnet"):
            property = name.split("-")[1]
            inner_dataset = MyMoleculeNetPos(property=property)
            idx_split = inner_dataset.get_idx_split(seed=seed)
            print(f"load {name} dataset with seed{seed}")
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
            return GraphormerPYGDataset(
                inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
            )
    
        elif name.startswith("ligand_binding_affinity"):
            subset = name.split("-")[1] # split_30 / split_60
            inner_dataset = None
            train_set = MyLBA(subset=subset, split="train")
            valid_set = MyLBA(subset=subset, split="valid")
            test_set = MyLBA(subset=subset, split="test")
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                ) if not qm9_data else GraphormerPYGDatasetQM9(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            data_func = GraphormerPYGDataset if not qm9_data else GraphormerPYGDatasetQM9
            return (
                None
                if inner_dataset is None
                else data_func(inner_dataset, seed, task_idx=task_idx)
            )
