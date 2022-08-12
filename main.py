from options import opt
from config import set_config
from modules.model import CNNMnist, ResNet9
from roles.FmpuTrainer import FmpuTrainer

import os
import torch


def main():
    set_config(opt)
    print("Acc from:", opt, "\n")

    if opt.dataset == 'MNIST':
        model = CNNMnist().cuda()
        trainer = FmpuTrainer(model)
    if opt.dataset == 'CIFAR10':
        model = ResNet9().cuda()
        trainer = FmpuTrainer(model)
    trainer.begin_train()



if __name__ == '__main__':
    # merge config
    main()