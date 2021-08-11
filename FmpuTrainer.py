from options import opt
from client import Client
from aggregator import Cloud
from dataSpilt import get_data_loaders
import numpy as np
import torch.optim as optim
from multiprocessing.dummy import Pool as ThreadPool
import copy
from loss import MPULoss
import matplotlib.pyplot as plt
import torch


class FmpuTrainer:
    def __init__(self, model_pu):
        # load data

        local_dataloaders, local_sample_sizes, test_dataloader , indexlist, priorlist = get_data_loaders()

        # create Clients and Aggregating Server
        self.clients = [Client(_id + 1, copy.deepcopy(model_pu).cuda(), local_dataloaders[_id], test_dataloader,  sample_size, opt.local_epochs,
                               opt.num_classes, priorList, indexList)
                            for sample_size, _id , priorList, indexList, in zip(local_sample_sizes, list(range(opt.num_clients)), priorlist, indexlist)]

        self.clientSelect_idxs = []
        # print(len(self.clients))
        self.cloud = Cloud(self.clients, model_pu, opt.num_classes, test_dataloader)
        self.communication_rounds = opt.communication_rounds
        self.current_round = 0


    def begin_train(self):
        print("Fmpu is going to train")
        # import pdb
        # pdb.set_trace()
        for t in range (self.communication_rounds):
            print("\n current round " + str(t)+"\n")

            self.current_round = t + 1
            self.clients_select()
            # client train step
            self.clients_train_step()   # memery up
            print("3:{}".format(torch.cuda.memory_allocated(0)))
            exit()


            self.clients_validation_step()
            w_glob = self.cloud.aggregate(self.clientSelect_idxs)


            for client in self.clients:
                client.model.load_state_dict(w_glob)
            #
            self.cloud.model.load_state_dict(w_glob)
            self.cloud.validation()


        # 所有clients重新初始化
        for client in self.clients:
            client.load_original_model()

        print("FL on Positive is going to train")
        FLAcc = []
        for t in range(self.communication_rounds):
            print("\n current round " + str(t) + "\n")
            self.current_round = t + 1
            self.clients_select()
            self.clients_train_step_P()
            # self.clients_validation_step()
            w_glob = self.cloud.aggregate(self.clientSelect_idxs)
            for client in self.clients:
                client.model.load_state_dict(w_glob)

            self.cloud.model.load_state_dict(w_glob)

        #     if t%10 == 0:
        #         FLAcc.append(self.cloud.validation())
        #
        # plotAcc(FmpuAcc, FLAcc)

    def clients_select(self):
        m = max(int(opt.clientSelect_Rate * opt.num_clients), 1)
        self.clientSelect_idxs = np.random.choice(range(opt.num_clients), m, replace=False)

    def clients_train_step(self):
        for idx in self.clientSelect_idxs:
            self.clients[idx].train_pu()

    def clients_train_step_P(self):
        for idx in self.clientSelect_idxs:
            self.clients[idx].train_P()

    def process(self, client):
        client.train_pu()

    def clients_validation_step(self): # 加载每一个model,在三个数据集上测试一下
        for client in self.clients:
            client.model



def plotAcc(FmpuAcc, FLAcc):
    epochs = list(range(len(FmpuAcc)))
    plt.plot(epochs, FmpuAcc, color='r', label='Train with PUloss Accuracy')  # r表示红色
    plt.plot(epochs, FLAcc, color='sandybrown', label='Train with Ploss Accuracy')  # r表示红色


    #####非必须内容#########
    plt.xlabel('communication rounds')  # x轴表示
    plt.ylabel('Accuracy')  # y轴表示
    plt.title("FLmpu Vs. FL_on_Positive")  # 图标标题表示
    plt.legend()  # 每条折线的label显示

    #plt.axhline(y=90.78, c='k', ls='--', lw=1)
    #plt.annotate(s='90.78%', xy=(0, 90.78), xytext=(0, 91))

    plt.ylim(0, 100)

    #######################
    plt.savefig(opt.imagename)  #T 保存图片，路径名为test.jpg
    plt.show()  # 显示图片


record_step = 1