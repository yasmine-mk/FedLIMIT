# from matplotlib.pyplot import step
# import torch
# from torch.utils.data import TensorDataset, Dataset
# from cnn import Model

# # from functools import partial
# from torch.optim import SGD
# from FedUtils.fed.client import Client

# def download_dataset(ds_name, root=None):
#     ds_name = ds_name.lower()
#     assert ds_name in ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist",
#      "tissuemnist", "organamnist", "organcmnist", "organsmnist"], "Dataset's name is not correct, please check the README for the available datasets"
#     import medmnist
#     from medmnist import INFO
#     dataclass = getattr(medmnist, INFO[ds_name]["python_class"])
#     if root == None:
#         train_dataset = dataclass(split="train", download=True)
#         test_dataset = dataclass(split="test", download=True)
#     else:
#         train_dataset = dataclass(root=root, split="train", download=True)
#         test_dataset = dataclass(root=root, split="test", download=True)
    
#     return train_dataset, test_dataset    
    

# def load_dataset(data_path):
#     # train_images val_images test_images train_labels val_labels test_labels
#     data = np.load(data_path)
#     train_dataset = TensorDataset(data["train_images"], data["train_labels"])
#     test_dataset = TensorDataset(data["test_images"], data["test_labels"])
#     return train_dataset, test_dataset    
    
# # def read_data(name="breastmnist",download=False,num_clients=5, train_data_path=None, test_data_path=None):

# import numpy as np


# def read_data(ds_name=None,download=False,num_clients=5, data_path=None, iid=True):
#     # print(ds_name)
#     # print(download)
#     # print(num_clients)
#     # print(data_path)
#     # print(iid)
#     def iid_partition(dataset, clients):
#         """
#         I.I.D paritioning of data over clients
#         Shuffle the data
#         Split it between clients

#         params:
#           - dataset (torch.utils.Dataset): Dataset containing the PathMNIST Images 
#           - clients (int): Number of Clients to split the data between
#         returns:
#           - Dictionary of image indexes for each client
#         """

#         num_items_per_client = int(len(dataset)/clients)
#         client_dict = {}
#         image_idxs = [i for i in range(len(dataset))]

#         for i in range(clients):
#             client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
#             image_idxs = list(set(image_idxs) - client_dict[i])
#         return client_dict # client dict has [idx: list(datapoint indices)
    
#     def non_iid_partition(dataset, num_clients):
#         """
#         non I.I.D parititioning of data over clients
#         Sort the data by the digit label
#         Divide the data into N shards of size S
#         Each of the clients will get X shards

#         params:
#           - dataset (torch.utils.Dataset): Dataset containing the pathMNIST Images
#           - num_clients (int): Number of Clients to split the data between
#           - total_shards (int): Number of shards to partition the data in
#           - shards_size (int): Size of each shard 
#           - num_shards_per_client (int): Number of shards of size shards_size that each client receives

#         returns:
#           - Dictionary of image indexes for each client
#         """
#         shards_size = 9
#         total_shards = len(dataset)// shards_size
#         num_shards_per_client = total_shards // num_clients
#         shard_idxs = [i for i in range(total_shards)]
#         client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
#         idxs = np.arange(len(dataset))
#         # get labels as a numpy array
#         data_labels = np.array([target.numpy().flatten() for _, target in dataset]).flatten()
#         # sort the labels
#         label_idxs = np.vstack((idxs, data_labels))
#         label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
#         idxs = label_idxs[0,:]

#         # divide the data into total_shards of size shards_size
#         # assign num_shards_per_client to each client
#         for i in range(num_clients):
#             rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
#             shard_idxs = list(set(shard_idxs) - rand_set)

#             for rand in rand_set:
#                 client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
#         return client_dict # client dict has [idx: list(datapoint indices)
    
#     #### 
#     assert not (download==False and data_path == None), "Either provide True for download or the data path"
#     assert not (download == True and ds_name == None),  "Provide a dataset name please"
    
#     # read_data(ds_name=None,download=False,num_clients=5, data_path=None)

#     if download:
#         train_dataset, test_dataset = download_dataset(ds_name=ds_name, root=data_path)
#         # print(type(train_dataset.imgs))
#         # print(type(train_dataset.labels))
#         train_dataset = TensorDataset(torch.Tensor(train_dataset.imgs), torch.tensor(train_dataset.labels))
#         test_dataset = TensorDataset(torch.Tensor(test_dataset.imgs), torch.tensor(test_dataset.labels))
#     else:
#         train_dataset, test_dataset = load_dataset(data_path)

#     if iid:
#         clients_dict = iid_partition(train_dataset, num_clients)
#     else:
#         clients_dict = non_iid_partition(train_dataset, num_clients)

#     # groups mean hierarchies, will not implement for now
#     groups = []

#     return train_dataset, clients_dict.keys(), groups, clients_dict, test_dataset


# class CustomDataset(Dataset):
#     def __init__(self, dataset, idxs, transform=None):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#         self.transforms = transform

#     def __len__(self):
#         return len(self.idxs)
    
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         if self.transforms is None:
#             image = torch.tensor(image)
#         else:
#             image = torch.tensor(self.transforms(image))
        
#         return image, torch.tensor(label)

# def step_func(model, data):
#     # lr = model.learning_rate
#     lr = 0.1
#     parameters = list(model.parameters())
#     flop = model.flop

#     def func(d):
#         nonlocal flop, lr
#         model.train()
#         model.zero_grad()
#         x, y = d
#         prediction = model.forward(x)
#         loss = model.loss(prediction, y).mean()
#         print(loss.item())
#         grad = torch.autograd.grad(loss, parameters)
#         for parameter, gradient in zip(parameters, grad):
#             parameter.data.add_(-lr*gradient)
#         return flop*len(x)
#     return func

# iid = True
# train_dataset, clients, groups, train_data, eval_data = read_data(ds_name="breastmnist", download=True, num_clients=1, data_path="./data", iid=iid)
# model = Model(num_classes=2,optimizer=SGD, learning_rate=0.1, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001)
# model_standalone = Model(num_classes=2,optimizer=SGD, learning_rate=0.1, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001)
# num_epochs = 3
# client = Client(id=0,
#                 group=1,
#                 train_data=train_data[0], 
#                 eval_data=eval_data, 
#                 model=model,batchsize=32, 
#                 train_transform=None, 
#                 test_transform=None, 
#                 traincusdataset=CustomDataset, 
#                 evalcusdataset=CustomDataset, 
#                 traindataset_all=train_dataset)
# # print("training the client")
# # client.solve_inner(num_epochs=10, step_func=None)
# # print("training te model on it's own")
# # model_standalone.solve_inner(data=client.train_data,num_epochs=10, step_func=step_func)
# import matplotlib.pyplot as plt

# # res = [-464.4102783203125, -5724.802734375, -5504.6181640625, -5284.43310546875, -5284.4326171875, -4403.6943359375, -4844.06396484375, -4844.06396484375, -6165.171875, -4403.6943359375, -5284.43310546875, -4844.06396484375, -4403.6943359375, -4623.87890625, -5504.6181640625, -4844.06396484375, -5724.80224609375, -7045.91064453125, -5724.802734375, -3963.324951171875, -4844.06396484375, -5284.43359375, -4183.50927734375, -5504.61767578125, -5284.43310546875, -4623.87890625, -4623.87890625, -5504.61767578125, -4844.0634765625, -4623.87890625, -6165.171875, -5284.43310546875, -5284.43310546875, -6385.3564453125, -5284.43310546875, -7045.91064453125, -5724.802734375, -4844.0634765625, -4844.06396484375, -5944.9873046875, -4403.6943359375, -5504.6181640625, -4844.06396484375, -5064.248046875, -6825.7255859375, -5724.80224609375, -4844.06396484375, -5724.802734375, -5504.61767578125, -3302.770751953125, -5504.6181640625, -4403.6943359375, -4403.6943359375, -7045.91064453125, -5944.9873046875, -5284.43359375, -3963.324951171875, -5284.43310546875, -5064.24853515625, -5284.43310546875, -5284.43310546875, -5724.802734375, -4844.0634765625, -5724.80224609375, -5064.24853515625, -5064.248046875, -5284.4326171875, -5504.61767578125, -4844.06396484375, -4623.87890625, -4623.87939453125, -7045.91064453125, -5944.98681640625, -5064.24853515625, -4623.87890625, -3743.14013671875, -6165.171875, -5944.9873046875, -5064.24853515625, -6165.171875, -4183.50927734375, -6385.3564453125, -5284.43310546875, -4403.6943359375, -4623.87890625, -4844.06396484375, -4403.6943359375, -5284.43359375, -5284.43310546875, -7045.91064453125, -5504.61767578125, -5724.80224609375, -5724.80224609375, -4403.6943359375, -4623.87890625, -5504.6181640625, -4403.6943359375, -5724.802734375, -5504.6181640625, -5284.43310546875, -5064.248046875, -4844.06396484375, -5504.61767578125, -4844.06396484375, -5284.43310546875, -4623.87939453125, -4844.06396484375, -7045.91064453125, -5064.24853515625, -4183.509765625, -4844.0634765625, -5944.9873046875, -5944.9873046875, -5064.248046875, -5284.43310546875, -4844.0634765625, -4623.87890625, -5504.61767578125, -4623.87939453125, -5504.61767578125, -5064.24853515625, -5504.61767578125, -5724.802734375, -5284.43310546875, -4623.87890625, -3522.955322265625, -4844.06396484375, -5284.43310546875, -5504.6181640625, -4844.06396484375, -4623.87890625, -5064.24853515625, -4403.6943359375, -5504.6181640625, -4403.6943359375, -5504.6181640625, -4844.06396484375, -5064.24853515625, -5064.24853515625, -5284.43310546875, -5724.802734375, -5284.43359375, -6385.3564453125, -3522.955322265625, -5064.24853515625, -5284.43310546875, -5944.9873046875, -5284.4326171875, -5724.80224609375, -4403.6943359375, -4403.6943359375, -5064.24853515625, -5504.6181640625, -4403.6943359375, -5284.43310546875, -5504.61767578125, -5064.24853515625, -5064.24853515625, -5064.248046875, -5064.24853515625, -5284.43310546875, -7045.91064453125, -3963.324951171875, -5944.9873046875, -5284.43310546875, -4183.50927734375, -5724.80224609375, -5504.61767578125, -5064.24853515625, -4403.6943359375, -4623.87890625, -5284.43359375, -5944.9873046875, -5284.43310546875, -5724.80224609375, -5064.24853515625, -4844.06396484375, -5284.43310546875, -5504.61767578125, -3522.955322265625]
# # plt.plot(range(len(res)), res)
# # plt.show()
# from torch.utils.data import DataLoader
# dataloader = DataLoader(train_dataset, batch_size=10)
# losses = []

# for i in range(100):
#     for data in dataloader:
#         _, loss = model_standalone.train_onestep(data)
#         losses.append(loss.item())

# plt.plot(range(len(losses)), losses)
# plt.show()

############################
from enum import unique
from logging import root
from cv2 import split, transform
from sklearn.utils import shuffle
from torch import nn
from FedUtils.models.utils import Flops, FSGM
import torch
import sys


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 576)


class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 784
        torch.manual_seed(seed)

        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5),
                                   nn.ReLU(), 
                                   nn.Conv2d(32, 32, 5), 
                                   nn.MaxPool2d(2), 
                                   nn.ReLU(), 
                                   nn.Conv2d(32, 64, 5),
                                   nn.MaxPool2d(2), 
                                   nn.ReLU(), 
                                   Reshape(), 
                                   nn.Linear(576, 256), 
                                   nn.ReLU(), 
                                   nn.Linear(256, self.num_classes)]
                                )
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)


        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), lr=0.1)
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate

        self.p_iters = p_iters
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta

        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.softmax(self.forward(x))

    def generate_fake(self, x, y):
        self.eval()
        psuedo, perturb = x.detach(), x.detach()
        if psuedo.device != next(self.parameters()).device:
            psuedo = psuedo.to(next(self.parameters()).device)
            perturb = perturb.to(next(self.parameters()).device)
        psuedo = FSGM(self, psuedo, y, self.p_iters, self.ps_eta)
        perturb = FSGM(self, perturb, y, self.p_iters, self.pt_eta)
        psuedo_y, perturb_y = self.predict(psuedo), self.predict(perturb)
        return [psuedo, y, psuedo_y], [perturb, y, perturb_y]

    def loss(self, pred, gt):
        pred = self.softmax(pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if len(gt.shape) != len(pred.shape):
            gt = nn.functional.one_hot(gt.long(), self.num_classes).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt*torch.log(pred+1e-12)
        loss = loss.sum(1)
        return loss

    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        return out

    def train_onestep(self, data):
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()
        x, y = data
        x, y = x.float(), y.float()
        pred = self.forward(x)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optimizer.step()
        return self.flop*len(x), loss

    def solve_inner(self, data, num_epochs=1, step_func=None):
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func is None:
            func = self.train_onestep
        else:
            func = step_func(self, data)

        for _ in range(num_epochs):
            for x, y in data:
                c = func([x, y])
                comp += c
                steps += 1.0
        soln = self.get_param()
        return soln, comp, weight

    def test(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        # print(f"data is of type {type(data)} and of length {len(data)}")
        # d is a tensor of shape (100,28,28)
        x, y = data
        with torch.no_grad():
            pred = self.forward(x)
        loss += self.loss(pred, y).sum()
        pred_max = pred.argmax(-1).unsqueeze(1).float()
        # print(pred_max.shape)
        # print(y.shape)
        assert len(pred_max.shape) == len(y.shape)
        if pred_max.device != y.device:
            pred_max = pred_max.detach().to(y.device)
        tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss

class altModel(nn.Module):
    def __init__(self, num_classes):
        super(altModel, self).__init__()
        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5),
                                   nn.ReLU(), 
                                   nn.Conv2d(32, 32, 5), 
                                   nn.MaxPool2d(2), 
                                   nn.ReLU(), 
                                   nn.Conv2d(32, 64, 5),
                                   nn.MaxPool2d(2), 
                                   nn.ReLU(), 
                                   Reshape(), 
                                   nn.Linear(576, 256), 
                                   nn.ReLU(), 
                                   nn.Linear(256, num_classes)]
                                )
    def forward(self, data):
        return self.net(data)
    


import medmnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import SGD
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn
# num_classes = 2
# model = Model(num_classes, optimizer=SGD, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001)
# num_epochs= 10
# losses = []
# print("training")
# model.train()
# lr = 0.1
# for i in range(num_epochs):
#     print(f"EPOCH {i+1}")
#     for data in tqdm(train_loader):
#         fp, loss = model.train_onestep(data)
#         losses.append(loss.item())

# plt.plot(losses)
# plt.show()
from torchvision.transforms import transforms
data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
train_dataset = getattr(medmnist, medmnist.INFO["organamnist"]["python_class"])(root="./data", split="train", transform=data_transform)
num_classes = len(np.unique(train_dataset.labels))
# plt.imshow(train_dataset[0][0])
# plt.show()
# x = np.array(train_dataset[:][0])
# y = np.array(train_dataset[:][1])
# print(x.shape)
# print(y.shape)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# model = altModel(2)
# optimizer  = SGD(model.parameters(), lr=0.1)
# criterion = nn.BCELoss()
# losses = []
# num_epochs = 10
# for i in range(num_epochs):
#     print(f"EPOCH {i+1}")
#     for imgs, labels in tqdm(train_loader):
#         imgs, labels = imgs.unsqueeze(1).float(), labels.flatten()
#         outputs = nn.Softmax(-1)(model(imgs))
#         print("outputs")
#         print(outputs)
#         print("labels")
#         print(labels)
#         loss = criterion(outputs, nn.functional.one_hot(labels))
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
import torch.optim as optim
import torch.nn.functional as F
losses = []
# Instantiate the model
model = altModel(num_classes=num_classes)

# Define the optimizer
optimizer = optim.Adam(model.parameters())

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the number of epochs to train for
num_epochs = 1
i = 0
# Iterate over the dataset for the specified number of epochs
model.train()
accs = []
for epoch in range(num_epochs):
    # Set the model to train mode
    print(f"epoch {epoch+1}")
    # Iterate over the batches in the dataset
    for data, labels in tqdm(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data.float())
        # Compute the loss
        loss = criterion(outputs, torch.argmax(labels, 1))
        if i == 0:
            print("this")
            print(outputs)
            print(torch.argmax(labels, 1))
            print(loss)
            print("end")
            i+=1
        # print(loss)
        # if loss.item()==0:
        #     exit()
        # Backward pass
        print("****************")
        print("labels",torch.argmax(labels, 1))
        print("outputs",torch.argmax(outputs, 1))
        print("****************")
        print(torch.argmax(outputs, 1) == torch.argmax(labels, 1))
        acc = np.sum(torch.argmax(outputs, 1) == torch.argmax(labels, 1))/ len(data)
        accs.append(acc)
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print the loss every 100 batches
        losses.append(loss.item())

plt.plot(losses)
plt.show()