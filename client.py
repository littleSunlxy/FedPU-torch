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


class Client:
    def __init__(self, client_id, model_pu, trainloader, testloader, samplesize, epoches, num_classes, priorlist, indexlist):
        self.client_id = client_id
        self.train_loader = trainloader
        self.test_loader = testloader
        self.current_round = 0
        self.batches = len(self.train_loader)
        self.samplesize = samplesize
        self.original_model = deepcopy(model_pu).cuda()
        self.model = model_pu
        if opt.positiveIndex == '0':
            self.loss = PLoss(num_classes).cuda()
        if opt.positiveIndex == 'randomIndexList':
            self.loss = MPULoss_INDEX(num_classes, opt.pu_weight).cuda()
        self.ploss = PLoss(num_classes)
        self.priorlist = priorlist
        self.indexlist = indexlist
        self.communicationRound = 0
        self.optimizer_pu = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_pu, step_size=1, gamma=0.992)
        self.optimizer_p = None
        self.scheduler_p = None
        print(self.client_id, self.samplesize)

    def load_original_model(self):
        self.model = deepcopy(self.original_model)
        self.communicationRound = 0
        self.optimizer_p = optim.SGD(self.model.parameters(), lr=opt.pu_lr, momentum=opt.momentum)
        self.scheduler_p = optim.lr_scheduler.StepLR(self.optimizer_p, step_size=1, gamma=0.992)


    def initialize(self):
        if os.path.exists(FedAVG_aggregated_model_path):
            self.model.load_state_dict(torch.load(FedAVG_aggregated_model_path))

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

                loss.backward()
                if i == 0:
                    print("loss:", loss, "ploss", ploss, "uloss", uloss)
                self.optimizer_pu.step()



        self.communicationRound+=1
        self.scheduler.step()



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

def plotAcc(trainPosAcc, trainAcc, testAcc, Ploss, Uloss):
    epochs = list(range(len(trainAcc)))
    plt.plot(epochs, trainAcc, color='r', label='Train Accuracy')  # r表示红色
    plt.plot(epochs, trainPosAcc, color='sandybrown', label='Train_Pos Accuracy')  # r表示红色
    plt.plot(epochs, testAcc, color='sienna', label='Test Accuracy')  # 蓝色
    plt.plot(epochs, Ploss, color='teal', linestyle='-', label='Positive Loss*5')  # r表示红色
    plt.plot(epochs, Uloss, color='c', linestyle='-', label='Unlabeled Loss*5')  # r表示红色


    #####非必须内容#########
    plt.xlabel('epochs')  # x轴表示
    plt.ylabel('Accuracy')  # y轴表示
    plt.title("MPU training using P and U")  # 图标标题表示
    plt.legend()  # 每条折线的label显示

    plt.axhline(y=90.78, c='k', ls='--', lw=1)

    plt.annotate(s='90.78%', xy=(0, 90.78), xytext=(0, 91))
    plt.ylim(0, 100)

    #######################
    plt.savefig('mpuAcc_from_00.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片



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


