from torch.utils.data import DataLoader
# to add, something like this, a dataset class that is generalizable to many data formats and type, npy, gyz etc...
# for automatic loading and other
# from FedUtils.models.utils import CusDataset
from collections import OrderedDict

class Client(object):
    # train_data = list of indexs
    # eval_data for now is a tensor dataset
    # traindataset_all= is the train_dataset.
    def __init__(self, id, group, train_data, eval_data, model, batchsize, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None, traindataset_all=None):
        super(Client, self).__init__()
        self.model = model
        self.id = id
        self.group = group
        # self.num_train_samples = len(train_data["x"])
        self.num_train_samples = len(train_data)
        # self.num_test_samples = [len(ed["x"]) for ed in eval_data]
        self.num_test_samples = len(eval_data)
        drop_last = False
        if traincusdataset:  # load data use customer's dataset
            # traincusdataset is CustomDataset from main
            # CustomDataset(dataset, idxs)
            self.train_data = DataLoader(traincusdataset(traindataset_all, train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            self.train_data_fortest = DataLoader(evalcusdataset(traindataset_all, [i for i in range(len(traindataset_all))], transform=test_transform), batch_size=batchsize, shuffle=False,)
            num_workers = 0
            self.eval_data = DataLoader(eval_data, batch_size=100, shuffle=False, num_workers=num_workers)
            # self.eval_data = [DataLoader(evalcusdataset(ed, transform=test_transform), batch_size=100, shuffle=False, num_workers=num_workers) for ed in eval_data]
        # else:
            # self.train_data = DataLoader(CusDataset(train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            # self.train_data_fortest = DataLoader(CusDataset(train_data, transform=test_transform), batch_size=batchsize, shuffle=False)
            # self.eval_data = [DataLoader(CusDataset(ed, transform=test_transform), batch_size=100, shuffle=False) for ed in eval_data]
        self.train_iter = iter(self.train_data)

    def set_param(self, state_dict) -> bool:
        '''
            set the local models parameters from a state dict
            returns true if it's successful
        '''
        self.model.set_param(state_dict)
        return True

    def get_param(self) -> OrderedDict[str, any]:
        '''
            get the local models parameters in a state_dict
        '''
        return self.model.get_param()

    def solve_grad(self):
        initial_model_size = self.model.size
        gradients, computation_cost = self.model.get_gradients(self.train_data)
        final_model_size = self.model.size
        training_stats = (self.num_train_samples, gradients)
        communication_stats = (initial_model_size, computation_cost, final_model_size)
        return training_stats, communication_stats
    
    def solve_inner(self, num_epochs=1, step_func=None):
        initial_model_size = self.model.size
        updated_model_params, computation_cost, updated_model_size = self.model.solve_inner(self.train_data, num_epochs=num_epochs, step_func=step_func)
        final_model_size = self.model.size
        training_stats = (self.num_train_samples * updated_model_size, updated_model_params)
        communication_stats = (initial_model_size, computation_cost, final_model_size)
        return training_stats, communication_stats

    def test(self):
        '''
        Executes the test function of the client's model on the validation data
        returns:
         * num_correct_samples
         * number of test samples
        '''
        correct_all = []
        # this is not used
        Loss = []
        for ed in self.eval_data:
            # ed is a list of two tensor elements
            # ed[0]: images shape
            # ed[1]: labels 
            # print(*[type(i) for i in ed])
            # print(*[i.shape for i in ed])
            # print(f"ed is of type {type(ed)} and of length {len(ed)}")
            total_correct, loss = self.model.test(ed)
            correct_all.append(total_correct)
            Loss.append(loss)
        return correct_all,  self.num_test_samples

    def train_error_and_loss(self):
        '''
        Executes the test function of the client's model on the training data
        returns:
         * num_correct_samples
         * loss
         * number of training samples
        '''
        # print("self.train_data_fortest", self.train_data_fortest)
        tot_correct = []
        loss = []

        for datapoint in self.train_data_fortest:
            correct , lss = self.model.test(datapoint)
            tot_correct.append(correct)
            loss.append(lss)
        #print("loss: ",lss)
        return sum(tot_correct), sum(loss), self.num_train_samples

    def __str__(self):
        return f'''
            model : {self.model}
            id : {self.id}
            group : {self.group}
            num_train_samples : {self.num_train_samples}
            num_test_samples : {self.num_test_samples}
        '''