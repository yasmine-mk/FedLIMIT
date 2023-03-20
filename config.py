from cnn import Model
from functools import partial
from FedUtils.fed.fedavg import FedAvg

config = {
    "dataset_name": "breastmnist",
    "seed": 11,  # random seed
    "data_path": "./data",
    "model": partial(Model, learning_rate=0.1, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001),  # the model to be trained
    "inner_opt": None,  # optimizer, in FedReg, only the learning rate is used
    "algorithm": FedAvg,  # Federated Learning algorithm, can be FedAvg, FedProx, FedBN or FedReg
    "iid": True,  # Wether we use IID data or non-IID data
    "num_classes": (2,),  # number of classes in the dataset [TODO] have it computed automatically.
    "clients_per_round": 10,  # number of clients sampled in each round
    "num_rounds": 3,  # number of total rounds
    "eval_every": 1,  # the number of rounds to evaluate the model performance. 1 is recommended here.
    "drop_percent": 0.0,  # the rate to drop a client.
    "num_epochs": 4,  # the number of epochs in local training stage
    "batch_size": 10,  # the batch size in local training stage
    "use_fed": 1,  # whether use federated learning alrogithms
    "log_path": "./logs/fedavg_cnn.log",  # the path to save the log file
    "train_transform": None,  # the preprocessing of train data, please refer to torchvision.transforms
    "test_transform": None,  # the preprocessing of test dasta
    "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
    "gamma": 0.4,  # the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used.
    "add_mask": -1
}