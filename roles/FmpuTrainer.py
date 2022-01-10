import numpy as np
import copy
import matplotlib.pyplot as plt
import torch

from datasets.dataSpilt import CustomImageDataset
from datasets.loader import DataLoader
from options import opt
from roles.client import Client
from roles.aggregator import Cloud
from datasets.dataSpilt import get_data_loaders, get_default_data_transforms
from modules.fedprox import GenerateLocalEpochs


class FmpuTrainer:
    def __init__(self, model_pu):
        # load data

        if not opt.useFedmatchDataLoader:
            # create Clients and Aggregating Server
            local_dataloaders, local_sample_sizes, test_dataloader , indexlist, priorlist = get_data_loaders()
            self.clients = [Client(_id + 1, copy.deepcopy(model_pu).cuda(), local_dataloaders[_id], test_dataloader,
                                   priorlist=priorList, indexlist=indexList)
                            for _id , priorList, indexList, in zip(list(range(opt.num_clients)), priorlist, indexlist)]
        else:
            self.loader = DataLoader(opt)
            # test_dataset = self.loader(get_test)
            # TODO: change to dataloader format
            indexlist = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 100
            priorlist = [[0.1] * 10] * 100
            self.load_data()
            self.loader.get_test()
            _, transforms_eval = get_default_data_transforms(opt.dataset, verbose=False)
            # test_dataset = CustomImageDataset(self.x_test, self.y_test, transforms_eval)
            test_dataset = CustomImageDataset(self.x_test.astype(np.float32)/255, self.y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batchsize, shuffle=True)

            self.clients = [Client(_id, copy.deepcopy(model_pu).cuda(), priorlist=priorList, indexlist=indexList)
                            for _id, priorList, indexList, in zip(list(range(opt.num_clients)), priorlist, indexlist)]
            print("numclients:", opt.num_clients, "build clients:", len(self.clients))

        self.clientSelect_idxs = []
        # print(len(self.clients))

        self.cloud = Cloud(self.clients, model_pu, opt.num_classes, test_dataloader)
        self.communication_rounds = opt.communication_rounds
        self.current_round = 0


    def load_data(self):
        # for Fedmatch dataloader
        self.x_train, self.y_train, self.task_name = None, None, None
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test, self.y_test =  self.loader.get_test()
        # self.x_test = self.loader.scale(self.x_test).transpose(0,3,1,2)
        self.x_test = self.x_test.transpose(0,3,1,2)
        self.y_test = torch.argmax(torch.from_numpy(self.y_test), -1).numpy()
        self.x_valid = self.loader.scale(self.x_valid)


    def begin_train(self):

        for t in range (self.communication_rounds):
            self.current_round = t + 1
            self.clients_select()
            # client train step
            if 'SL' in opt.method:
                print("##### FedAvg SL is training #####")
                self.clients_train_step_P()
            else:
                print("##### FedPU is training #####")
                self.clients_train_step()   # memery up

            # self.clients_validation_step()
            w_glob = self.cloud.aggregate(self.clientSelect_idxs)

            for client in self.clients:
                # client.model.load_state_dict(w_glob)
                client.mode = copy.deepcopy(w_glob)

            self.cloud.model.load_state_dict(w_glob)
            self.cloud.validation(self.current_round)


    def clients_select(self):
        m = max(int(opt.clientSelect_Rate * opt.num_clients), 1)
        self.clientSelect_idxs = np.random.choice(range(opt.num_clients), m, replace=False)

    def clients_train_step(self):
        if 'FedProx' in opt.method:
            percentage = opt.percentage    # 0.5  0.9
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clients)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients), max_epochs=opt.FedProx_Epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)

            for idx in self.clientSelect_idxs:
                if opt.usePU:
                    self.clients[idx].train_fedprox_pu(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
                else:
                    self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
        else:
            for idx in self.clientSelect_idxs:
                self.clients[idx].train_pu()


    def clients_train_step_P(self):
        for idx in self.clientSelect_idxs:
            self.clients[idx].train_P()
