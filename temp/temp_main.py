import argparse
import importlib
from multiprocessing.connection import Client
from random import random
import numpy as np
import torch
import json
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
warnings.filterwarnings("ignore")

def download_dataset(ds_name, root=None):
    ds_name = ds_name.lower()
    assert ds_name in ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist",
     "tissuemnist", "organamnist", "organcmnist", "organsmnist"], "Dataset's name is not correct, please check the README for the available datasets"
    import medmnist
    from medmnist import INFO
    dataclass = getattr(medmnist, INFO[ds_name]["python_class"])
    if root == None:
        train_dataset = dataclass(split="train", download=True)
        test_dataset = dataclass(split="test", download=True)
    else:
        train_dataset = dataclass(root=root, split="train", download=True)
        test_dataset = dataclass(root=root, split="test", download=True)
    
    return train_dataset, test_dataset    
    