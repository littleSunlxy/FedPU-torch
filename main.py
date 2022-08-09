from options import opt
from config import set_config
from modules.model import CNNMnist, ResNet9
from roles.FmpuTrainer import FmpuTrainer

import os



def main():
    set_config(opt)
    print("Acc from:", opt, "\n")
    if opt.dataset == 'MNIST':
        trainer = FmpuTrainer(CNNMnist().cuda())
    if opt.dataset == 'CIFAR10':
        trainer = FmpuTrainer(ResNet9().cuda())
        #trainer = FmpuTrainer(CNNCifar(opt).cuda())
    # trainer = FmpuTrainer(MLP(dim_in=784, dim_hidden=200, dim_out=10).cuda())
    # m_state_dict = torch.load('/home/linxinyang/mpuFL2/fed.pt')
    # trainer.cloud.model.load_state_dict(m_state_dict)
    # trainer.cloud.model.eval()
    # trainer.cloud.validation()
    trainer.begin_train()



if __name__ == '__main__':
    # merge config
    main()