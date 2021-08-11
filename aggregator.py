
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
        self.aggregated_client_model = {}

    def aggregate(self, clientSelect_idxs):
        w_locals = []
        weight = []
        positive_totalsize = 0
        totalsize = 0
        for idx in clientSelect_idxs:
            client = self.clients[idx]
            totalsize += client.samplesize
        #
        # for idx in clientSelect_idxs:
        #     client = self.clients[idx]
        #     weight.append(client.samplesize / totalsize)
        #     w = client.model.state_dict()
        #     w_locals.append(copy.deepcopy(w))
        #
        # w_avg = copy.deepcopy(w_locals[0])
        # for k in w_avg.keys():
        #     print(k)
        #     w_avg[k] *= weight[0]
        #     for i in range(1, len(w_locals)):
        #         w_avg[k] += w_locals[i][k] * weight[i]
        #     #w_avg[k] = torch.div(w_avg[k], len(w_locals))

        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx]
            weight = client.samplesize / totalsize
            # print(client.client_id, client.sample_size, self.total_client_data_size, weight)
            for name, param in client.model.state_dict().items():
                if k == 0:
                    self.aggregated_client_model[name] = param.data * weight
                else:
                    self.aggregated_client_model[name] += param.data * weight

        return self.aggregated_client_model

    def validation(self):
        self.model.eval()
        correct = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            correct += (pred == labels).sum().item()
        print('Accuracy of the global model on the testing sets: {:.4f} %'.format(100 * correct / len(self.test_loader.dataset)))
        return 100 * correct / len(self.test_loader.dataset)



    def _save_model(self):
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)

