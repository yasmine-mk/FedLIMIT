from .client import Client
from collections import OrderedDict
import numpy as np
class Server(object):
    def __init__(self, config: dict, model, train_dataset_all, datasets, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None):
        super(Server, self).__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.algorithm = config["algorithm"]
        self.clients_per_round = config["clients_per_round"]
        self.eval_every = config["eval_every"]
        self.num_rounds = config["num_rounds"]
        self.batch_size = config["batch_size"]
        # To simulate system heterogeneity
        self.drop_percent = config["drop_percent"]
        # Number of local epochs
        self.num_epochs = config["num_epochs"]
        self.eval_train = config["eval_train"]
        self.gamma = config["gamma"] if "gamma" in config else 1.0
        self.add_mask = config["add_mask"]
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.traincusdataset = traincusdataset
        self.evalcusdataset = evalcusdataset


        self.train_dataset = train_dataset_all
        # ommited this for testing as we don't have the model class yet
        # self.model = model()
        self.model = model(*self.num_classes, config["inner_opt"])
        self.clients_model = model(*self.num_classes, config["inner_opt"])
        self.clients = self.make_clients(datasets, model)

    def make_clients(self, datasets, Model):
        users, groups, train_data_idxs, test_data = datasets
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = list()
        for user, group in zip(users, groups):
            # make it work with our read_data function, 
            # each user will have a copy of test_dataset
            all_clients.append((user, group, train_data_idxs[user], test_data, Model , self.batch_size, self.train_transform, self.test_transform, self.train_dataset))
        # all_clients = [(u, g, train_data[u], [td[u] for td in test_data], Model
        # , self.batch_size, self.train_transform, self.test_transform) for u, g in zip(users, groups)]
        return all_clients
    
    def select_clients(self, seed, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(seed)
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        clients = [self.clients[c] for c in indices]
        clients = [Client(c[0], c[1], c[2], c[3], self.clients_model, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset, self.train_dataset) for c in clients]
        return indices, clients

    def set_param(self, state_dict) -> bool:
        '''
            alias of model.load_state_dict()
        '''

        self.model.set_param(state_dict)
        return True

    def get_param(self) -> dict[str, any]:
        '''
            return model.state_dict()
        '''
        return self.model.state_dict()
        return self.model.get_param()

    def _aggregate(self, wstate_dicts) -> dict[str, any]:
        '''
            aggregates mutiple state dicts
            returns an aggregated state dict
        '''
        old_params = self.get_param()
        state_dict = {x: 0.0 for x in self.get_param()}
        wtotal = 0.0
        for w, st in wstate_dicts:
            wtotal += w
            for name in state_dict.keys():
                assert name in state_dict
                state_dict[name] += st[name]*w
        state_dict = {x: state_dict[x]/wtotal for x in state_dict}
        return state_dict

    def _aggregate(self, wstate_dicts) -> OrderedDict[str, any]:
        '''
            aggregates mutiple state dicts
            returns an aggregated state dict
        '''

        state_dict_final = {x: 0.0 for x in self.get_param()}
        wtotal = 0.0
        for weight, state_dict in wstate_dicts:
            wtotal += weight
            for name in state_dict_final.keys():
                assert name in state_dict_final
                state_dict_final[name] += state_dict[name]*weight
        state_dict_final = {x: state_dict_final[x]/wtotal for x in state_dict_final}
        return state_dict_final

    def aggregate(self, weighted_state_dicts) -> bool:
        '''
            Aggregates multiple state dicts and then sets the aggregated(averaged) models' state parameters to
            self.model using self.set_params(state_dict)
            uses self._aggregate(wstate_dicts)
        '''
        state_dict = self._aggregate(weighted_state_dicts)
        return self.set_param(state_dict)
    
    def train(self):
        print("training")
        raise NotImplementedError
        
    def save(self):
        raise NotImplementedError

    # the return type might not be correct
    def test(self) -> tuple[list[int], list[int], list[list[int]], list[list[int]]]:
        '''
            Test the server on clients who have test data
            returns ids, groups, num_samples, total_correct
            accuracy can be calculated by = total_correct / num_samples 
        '''
        num_samples = []
        total_correct = []
        # keep an eye on the hard coded value [3][0]['x'], because we might change the data structure so images may not be selected using 'x'
        # (user, group, train_data_idxs[user], test_data, Model , self.batch_size, self.train_transform, self.test_transform, self.train_dataset)
        clients = [client for client in self.clients if len(client[3]) > 0]
        #Client init header:
        #self, id, group, train_data, eval_data, model, batchsize, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None, traindataset_all=None)

        clients = [Client(c[0], c[1], c[2], c[3], self.clients_model, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset, self.train_dataset) for c in clients]
        [client.set_param(self.get_param()) for client in clients]

        for client in clients:
            correct, number_samples_client = client.test()
            total_correct.append(correct)
            num_samples.append(number_samples_client)
        
        ids = [client.id for client in clients]
        groups = [client.group for client in clients]
        num_test = len(total_correct[0])
        # very confusing how they do the testing here.
        tot_correct = [sum(i) for i in total_correct]
        # num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.clients
        clients = [Client(c[0], c[1], c[2], c[3], self.clients_model, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset, self.train_dataset) for c in clients]
        [m.set_param(self.get_param()) for m in clients]
        for c in clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses

    def __str__(self):
        return f'''Server of {len(self.clients)} clients
            * model: {self.model.name if isinstance(self.model, str) else "model not named, add name attribute to the model"}
            * num_classes: {self.num_classes[0]}
            * algorithm: {self.algorithm}
            * clients_per_round: {self.clients_per_round}
            * num_rounds: {self.num_rounds}
            * evaluate_every: {self.evaluate_every}
            * batch_size: {self.batch_size}
            * drop_percent: {self.drop_percent}
            * num_epochs: {self.num_epochs}
            * eval_train: {self.eval_train}
            * gamma: {self.gamma}
            * add_mask: {True if self.add_mask > 0 else False}
            * train_tranform: {self.train_transform}
            * test_transform: {self.test_transform}
        '''