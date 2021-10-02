"""
client in Fmpu
"""
import time
import torch
from torch import nn
from copy import deepcopy
import torch.optim as optim
from pylab import *
import matplotlib.pyplot as plt
from options import opt
from options import FedAVG_model_path, FedAVG_aggregated_model_path
from loss import MPULoss, PLoss, MPULoss_INDEX
from datasets.loader import DataLoader


class Client:
    def __init__(self, client_id, model_pu, trainloader=None, testloader=None, priorlist=None, indexlist=None):
        self.client_id = client_id
        self.current_round = 0
        self.batches = opt.pu_batchsize
        self.original_model = deepcopy(model_pu).cuda()
        self.model = model_pu
        if opt.positiveIndex == '0':
            self.loss = PLoss(opt.num_classes).cuda()
        if opt.positiveIndex == 'randomIndexList':
            self.loss = MPULoss_INDEX(opt.num_classes, opt.pu_weight).cuda()
        self.ploss = PLoss(opt.num_classes)
        self.priorlist = priorlist
        self.indexlist = indexlist
        self.communicationRound = 0
        self.optimizer_pu = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_pu, step_size=1, gamma=0.992)
        self.optimizer_p = None
        self.scheduler_p = None

        if not opt.useFedmatchDataLoader:
            self.train_loader = trainloader
            self.test_loader = testloader
        else:
            # for Fedmatch
            self.state = {'client_id': client_id}
            self.loader = DataLoader(opt)
            self.load_data()

    def load_original_model(self):
        self.model = deepcopy(self.original_model)
        self.communicationRound = 0
        self.optimizer_p = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler_p = optim.lr_scheduler.StepLR(self.optimizer_p, step_size=1, gamma=0.992)


    def initialize(self):
        if os.path.exists(FedAVG_aggregated_model_path):
            self.model.load_state_dict(torch.load(FedAVG_aggregated_model_path))


    #----------------use FedMatch dataloader------------

    def load_data(self):
        self.x_labeled, self.y_labeled, task_name = \
                self.loader.get_s_by_id(self.state['client_id'])
        self.x_unlabeled, self.y_unlabeled, task_name = \
                self.loader.get_u_by_id(self.state['client_id'], task_id=0)
        self.x_test, self.y_test =  self.loader.get_test()
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid)

    #----------------use FedMatch dataloader------------



    def train_pu(self):
        self.model.train()
        for epoch in range(opt.local_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_pu.zero_grad()  # tidings清零
                outputs = self.model(inputs)  # on cuda 0
                # print(outputs.dtype, outputs.device)

                if opt.positiveIndex == '0':
                    loss = self.loss(outputs, labels)
                if opt.positiveIndex == 'randomIndexList':
                    loss, ploss, uloss = self.loss(outputs, labels, self.priorlist, self.indexlist)
                # print("lr:", self.optimizer_pu.param_groups[-1]['lr'])
                loss.backward()
                # if i == 0:
                #     print("epoch", epoch, "loss:", loss, "ploss", ploss, "uloss", uloss)
                self.optimizer_pu.step()

        self.communicationRound+=1
        self.scheduler.step()


    def train_P_fedmatch(self):
        bsize_s = opt.bsize_s
        num_steps = round(len(self.x_labeled)/bsize_s)
        bsize_u = math.ceil(len(self.x_unlabeled)/max(num_steps,1))  # 101

        x_labeled = self.x_labeled[i*bsize_s:(i+1)*bsize_s]
        y_labeled = self.y_labeled[i*bsize_s:(i+1)*bsize_s]
        x_unlabeled = self.x_unlabeled[i*bsize_u:(i+1)*bsize_u]
        y_unlabeled = self.x_unlabeled[i*bsize_u:(i+1)*bsize_u]

        # merge to new trainloader



    def train_P(self):
        self.model.train()
        for epoch in range(opt.local_epochs):

            for i, (inputs, labels) in enumerate(self.train_loader):

                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer_p.zero_grad()  # tidings清零
                outputs = self.model(inputs)
                loss = self.ploss(outputs, labels)
                loss.backward()
                self.optimizer_p.step()

        self.communicationRound += 1
        self.scheduler_p.step()



    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs)
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            total += pred.size(0)
            correct += (pred == labels).sum().item()
        print('Accuracy of the {} on the testing sets: {:.4f} %%'.format(self.client_id, 100 * correct / total))
        return 100 * correct / total



    def cal_trainAcc(self):
        self.model.eval()
        correct = 0
        total = 0
        # random choose
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs)
            orilabels = labels - opt.num_classes
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            correct += (pred == orilabels).sum().item()
            break
        print('Accuracy of the {} on the training sets: {:.4f} %%'.format(self.client_id, 100 * correct / total))
        return 100.0 * correct / total



def cal_train_PositveAcc(model, num_classes, images, labels):
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        outputs = model(images)
        mask = (labels < num_classes).nonzero().view(-1)
        groundTruth = torch.index_select(labels, 0, mask)
        predictOnLabeled = torch.index_select(outputs, 0, mask)
        pred = predictOnLabeled.data.max(1, keepdim=True)[1].view(groundTruth.shape[0]).cuda()
        total += pred.size(0)
        correct += (pred == groundTruth).sum().item()
    print('Accuracy of the network on the training 【Positive】 sets: {:.4f} %%' .format(100 * correct / total))
    return 100.0 * correct / total


