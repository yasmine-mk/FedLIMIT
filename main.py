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


# def read_data(train_data_path, test_data_path):
#     if not isinstance(test_data_path, list):
#         test_data_path = [test_data_path, ]
#     groups = []
#     train_data = {}
#     test_data = [{} for _ in test_data_path]
#     train_files = os.listdir(train_data_path)
#     train_files = [f for f in train_files if f.endswith(".json")]
#     for f in train_files:
#         file_path = os.path.join(train_data_path, f)
#         with open(file_path, "r") as inf:
#             cdata = json.load(inf)
#         if "hierarchies" in cdata:
#             groups.extend(cdata["hierarchies"])
#         train_data.update(cdata["user_data"])
#     for F, td in zip(test_data_path, test_data):
#         test_files = os.listdir(F)
#         test_files = [f for f in test_files if f.endswith(".json")]
#         for f in test_files:
#             file_path = os.path.join(F, f)
#             with open(file_path, "r") as inf:
#                 cdata = json.load(inf)
#             td.update(cdata["user_data"])
#     clients = list(sorted(train_data.keys()))
#     return clients, groups, train_data, test_data



# What we need now for the data, we need a function that partitions the data obtained from the npz file into IID and non IID
# Two way to do this, either we directly do it at first, meaning at the first stage we do the partitioning of data into somekind of a
# data structure or a class and then we work on it or we keep propagating the parameters of the partitioning across the program or something.

# okay lets start, the file we'll be dealing with for medmnist datasets is .npz file:
#   Which is a dict of keys:
#   * train_images.
#   * val_images.
#   * test_images.
#   * train_labels.
#   * val_labels.
#   * test_labels.
#   And all the entries in the dict are of type np.ndarray
# [to do] iid and non iid data partitioning
# data function downloading and loading function

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
    

def load_dataset(data_path):
    # train_images val_images test_images train_labels val_labels test_labels
    data = np.load(data_path)
    train_dataset = TensorDataset(data["train_images"], data["train_labels"])
    test_dataset = TensorDataset(data["test_images"], data["test_labels"])
    return train_dataset, test_dataset    
    
# def read_data(name="breastmnist",download=False,num_clients=5, train_data_path=None, test_data_path=None):

import numpy as np


def read_data(ds_name=None,download=False,num_clients=5, data_path=None, iid=True):
    # print(ds_name)
    # print(download)
    # print(num_clients)
    # print(data_path)
    # print(iid)
    def iid_partition(dataset, clients):
        """
        I.I.D paritioning of data over clients
        Shuffle the data
        Split it between clients

        params:
          - dataset (torch.utils.Dataset): Dataset containing the PathMNIST Images 
          - clients (int): Number of Clients to split the data between
        returns:
          - Dictionary of image indexes for each client
        """

        num_items_per_client = int(len(dataset)/clients)
        client_dict = {}
        image_idxs = [i for i in range(len(dataset))]

        for i in range(clients):
            client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
            image_idxs = list(set(image_idxs) - client_dict[i])
        return client_dict # client dict has [idx: list(datapoint indices)
    
    def non_iid_partition(dataset, num_clients):
        """
        non I.I.D parititioning of data over clients
        Sort the data by the digit label
        Divide the data into N shards of size S
        Each of the clients will get X shards

        params:
          - dataset (torch.utils.Dataset): Dataset containing the pathMNIST Images
          - num_clients (int): Number of Clients to split the data between
          - total_shards (int): Number of shards to partition the data in
          - shards_size (int): Size of each shard 
          - num_shards_per_client (int): Number of shards of size shards_size that each client receives

        returns:
          - Dictionary of image indexes for each client
        """
        shards_size = 9
        total_shards = len(dataset)// shards_size
        num_shards_per_client = total_shards // num_clients
        shard_idxs = [i for i in range(total_shards)]
        client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
        idxs = np.arange(len(dataset))
        # get labels as a numpy array
        data_labels = np.array([target.numpy().flatten() for _, target in dataset]).flatten()
        # sort the labels
        label_idxs = np.vstack((idxs, data_labels))
        label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
        idxs = label_idxs[0,:]

        # divide the data into total_shards of size shards_size
        # assign num_shards_per_client to each client
        for i in range(num_clients):
            rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
            shard_idxs = list(set(shard_idxs) - rand_set)

            for rand in rand_set:
                client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
        return client_dict # client dict has [idx: list(datapoint indices)
    
    #### 
    assert not (download==False and data_path == None), "Either provide True for download or the data path"
    assert not (download == True and ds_name == None),  "Provide a dataset name please"
    
    # read_data(ds_name=None,download=False,num_clients=5, data_path=None)

    if download:
        train_dataset, test_dataset = download_dataset(ds_name=ds_name, root=data_path)
        # print(type(train_dataset.imgs))
        # print(type(train_dataset.labels))
        train_dataset = TensorDataset(torch.Tensor(train_dataset.imgs), torch.tensor(train_dataset.labels))
        test_dataset = TensorDataset(torch.Tensor(test_dataset.imgs), torch.tensor(test_dataset.labels))
    else:
        train_dataset, test_dataset = load_dataset(data_path)

    if iid:
        clients_dict = iid_partition(train_dataset, num_clients)
    else:
        clients_dict = non_iid_partition(train_dataset, num_clients)

    # groups mean hierarchies, will not implement for now
    groups = []

    return train_dataset, clients_dict.keys(), groups, clients_dict, test_dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transforms = transform

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transforms is None:
            image = torch.tensor(image)
        else:
            image = torch.tensor(self.transforms(image))
        
        return image, torch.tensor(label)

# from FedUtils.fed.server import Server
# from FedUtils.fed.fedavg import FedAvg

def main():
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("-c", "--config", help="path to the config file")
    arguments = argumentparser.parse_args()
    config = importlib.import_module(arguments.config.replace("/","."),package=None)
    config = config.config
    
    # from FedUtils.models.resnet9 import resnet9
    class logger:
        def __init__(self) -> None:
            pass
    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # [to do] add automatic models from just the name later
    Model = config["model"]
    iid = config["iid"]
    # to be modified once we've figured out the dataset format
    # everything to about datasets we'll skip for now
    # clients: list of client indexs
    # groups: empty list
    # train_data: dict[users: list[datapoint indxs]]
    # eval_data: TensorDataset.
    train_dataset, clients, groups, train_data, eval_data = read_data(ds_name=config["dataset_name"], download=True, num_clients=10, data_path=config["data_path"], iid=iid)
    Dataset = CustomDataset


    if config["use_fed"]:
        algorithm = config["algorithm"]
        t = algorithm(config=config, model=Model, train_dataset_all=train_dataset, datasets=[clients, groups, train_data, eval_data], train_transform=config["train_transform"],
                      test_transform=config['test_transform'], traincusdataset=Dataset, evalcusdataset=Dataset)
        t.train()
        print("finished")
    # this doesn't work because we changed the data format for now
########################
    else:
        train_data_total = {"x": [], "y": []}
        eval_data_total = {"x": [], "y": []}
        for t in train_data:
            train_data_total["x"].extend(train_data[t]["x"])
            train_data_total["y"].extend(train_data[t]["y"])
        for t in eval_data:
            eval_data_total["x"].extend(eval_data[t]["x"])
            eval_data_total["y"].extend(eval_data[t]["y"])
        train_data_size = len(train_data_total["x"])
        eval_data_size = len(eval_data_total["x"])
        train_data_total_fortest = DataLoader(Dataset(train_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        train_data_total = DataLoader(Dataset(train_data_total, config["train_transform"]), batch_size=config["batch_size"], shuffle=True, )
        eval_data_total = DataLoader(Dataset(eval_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        # model = Model(*config["model_param"], optimizer=inner_opt)
        model = Model(*config["model_param"], optimizer=config["inner_opt"])
        for r in range(config["num_rounds"]):
            model.solve_inner(train_data_total)
            stats = model.test(eval_data_total)
            train_stats = model.test(train_data_total_fortest)
            logger.info("-- Log At Round {} --".format(r))
            logger.info("-- TEST RESULTS --")
            logger.info("Accuracy: {}".format(stats[0]*1.0/eval_data_size))
            logger.info("-- TRAIN RESULTS --")
            logger.info(
                "Accuracy: {} Loss: {}".format(train_stats[0]/train_data_size, train_stats[1]/train_data_size))

if __name__=='__main__':
    main()
