
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
        totalsize = 0
        samplesize = 500
        for idx in clientSelect_idxs:
            totalsize += samplesize

        global_model = {}
        model_dict = self.aggregated_client_model.state_dict()
        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx]
            weight = samplesize / totalsize
            for name, param in client.model.state_dict().items():
                if k == 0:
                    global_model[name] = param.data * weight
                else:
                    global_model[name] += param.data * weight

        pretrained_dict = {k: v for k, v in global_model.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        import pdb;
        pdb.set_trace()
        model_dict.update(pretrained_dict)
        self.aggregated_client_model.load_state_dict(pretrained_dict)

        # updating the global weights
        # import pdb;
        # pdb.set_trace()
        # self.aggregated_client_model = self.model
        # # import pdb; pdb.set_trace()
        # weights_avg = copy.deepcopy(self.clients[clientSelect_idxs[0]].model)
        # # print("res2.1.0.bias before:", weights_avg.state_dict()['res2.1.0.bias'].sum())
        # for k in weights_avg.state_dict().keys():
        #     for index, i in enumerate(clientSelect_idxs):
        #         weights_avg.state_dict()[k] += self.clients[i].model.state_dict()[k]
        #         print(weights_avg.state_dict()[k].sum())
        #     import pdb;
        #     pdb.set_trace()
        #     weights_avg.state_dict()[k] = torch.div(weights_avg.state_dict()[k], len(clientSelect_idxs))
        #     weights_avg.state_dict().update(k, torch.div(weights_avg.state_dict()[k], len(clientSelect_idxs)))
        #     print("\n after div ---sum:", weights_avg.state_dict()[k].sum())
        #
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(pretrained_dict)
        # # print("res2.1.0.bias after:", weights_avg.state_dict()['res2.1.0.bias'].sum())
        #
        # self.aggregated_client_model = weights_avg
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

