
"""
Cloud server
"""
import torch
import copy
import torch.nn as nn
from options import FedAVG_model_path, FedAVG_aggregated_model_path
from torch.utils.data import DataLoader


class Cloud:
    def __init__(self, clients, model, numclasses, dataloader):
        self.model = model
        self._save_model()
        self.clients = clients
        self.numclasses = numclasses
        self.test_loader = dataloader
        self.participating_clients = None
        self.aggregated_client_model = model

    def aggregate(self, clientSelect_idxs):
        # totalsize = 0
        # samplesize = 500
        # # for idx in clientSelect_idxs:
        #     totalsize += samplesize

        # for k, idx in enumerate(clientSelect_idxs):
        #     client = self.clients[idx]
        #     weight = samplesize / totalsize
        #     if self.aggregated_client_model == None:
        #         self.aggregated_client_model = {}
        #     for name, param in client.model.state_dict().items():
        #         if k == 0:
        #             import pdb; pdb.set_trace()
        #             self.aggregated_client_model[name] = param.data * weight
        #         else:
        #             self.aggregated_client_model[name] += param.data * weight

        # updating the global weights
        # import pdb;
        # pdb.set_trace()
        self.aggregated_client_model = self.model
        import pdb; pdb.set_trace()
        weights_avg = copy.deepcopy(self.clients[0].model)
        print("res2.1.0.bias before:", weights_avg.state_dict()['res2.1.0.bias'].sum())
        for k in weights_avg.state_dict().keys():
            for index, i in enumerate(clientSelect_idxs):
                weights_avg.state_dict()[k] += self.clients[i].model.state_dict()[k]
            weights_avg.state_dict()[k] = torch.div(weights_avg.state_dict()[k], len(clientSelect_idxs))

        print("res2.1.0.bias after:", weights_avg.state_dict()['res2.1.0.bias'].sum())
        import pdb;
        pdb.set_trace()
        self.aggregated_client_model = weights_avg
        return self.aggregated_client_model

    def validation(self, cur_rounds):
        self.aggregated_client_model.eval()
        correct = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            # print("Test input img scale:", inputs.max(), inputs.min())
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.aggregated_client_model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            correct += (pred == labels).sum().item()
        print('Round:{:d}, Accuracy: {:.4f} %'.format(cur_rounds, 100 * correct / len(self.test_loader.dataset)))
        return 100 * correct / len(self.test_loader.dataset)



    def _save_model(self):
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)

