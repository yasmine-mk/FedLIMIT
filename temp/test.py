from FedUtils.fed.server import Server
import numpy as np
import torch
from torch.utils.data import TensorDataset
from FedUtils.models.resnet9 import resnet9
from functools import partial
#_(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):

config = {
    "dataset_name": "breastmnist",
    "seed": 1,  # random seed
    "model": partial(resnet9, learning_rate=0.1, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001),  # the model to be trained
    "inner_opt": None,  # optimizer, in FedReg, only the learning rate is used
    "algorithm": None,  # FL optimizer, can be FedAvg, FedProx, FedCurv or SCAFFOLD
    "num_classes": (10,),  # the input of the model, used to initialize the model
    "inp_size": (784,),  # the input shape
    "train_path": "data/mnist_10000/data/train/",  # the path to the train data
    "test_path": "data/mnist_10000/data/valid/",  # the path to the test data
    "clients_per_round": 10,  # number of clients sampled in each round
    "num_rounds": 500,  # number of total rounds
    "eval_every": 1,  # the number of rounds to evaluate the model performance. 1 is recommend here.
    "drop_percent": 0.0,  # the rate to drop a client. 0 is used in our experiments
    "num_epochs": 40,  # the number of epochs in local training stage
    "batch_size": 10,  # the batch size in local training stage
    "use_fed": 1,  # whether use federated learning alrogithms
    "log_path": "tasks/mnist/FedReg/train.log",  # the path to save the log file
    "train_transform": None,  # the preprocessing of train data, please refer to torchvision.transforms
    "test_transform": None,  # the preprocessing of test dasta
    "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
    "gamma": 0.4,  # the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used, or the value of lambda when FedCurv is used
    "add_mask" : 0,
    "iid": False
}


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


def read_data(ds_name=None,download=False,num_clients=5, data_path=None, iid=True):
    print(ds_name)
    print(download)
    print(num_clients)
    print(data_path)
    print(iid)
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
        print(type(train_dataset.imgs))
        print(type(train_dataset.labels))
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

iid = config["iid"]
train_dataset, clients, groups, train_data, eval_data = read_data(ds_name="organamnist", download=True, num_clients=10, data_path="./data", iid=iid)

server = Server(config, config["model"], train_dataset, [clients, groups, train_data, eval_data], train_transform=None, test_transform=None)
print(server)
print(server.get_param().keys())
print(type(server.get_param()))
