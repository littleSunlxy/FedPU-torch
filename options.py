
import argparse

parser = argparse.ArgumentParser()
# optimizer
parser.add_argument('--pu_lr', type=float, default=0.01, help='learning rate of each client')
parser.add_argument('--adjust_lr', action='store_true', default=False,
                    help='adjust lr according to communication rounds')
parser.add_argument('--pu_batchsize', type=int, default=500, help='batchsize of dataloader')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer param')
# dataset
# parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--num_classes', type=int, default=10)
# PU param
parser.add_argument('--pu_weight', type=float, default=1, help='weight of puloss') #1
parser.add_argument('--local_epochs', type=int, default=20, help='epoches of each client')
parser.add_argument('--use_PULoss', action='store_true', default=False,
                    help='use PULoss or only PLoss')
# pu dataloader                  
parser.add_argument('--is_noniid', action='store_true', default=False,
                    help='non-iid setting')
parser.add_argument('--randomIndex_num', type=int, default=2,
help='rate of positive sample')
parser.add_argument('--P_Index_accordance', action='store_true', 
                    help='the same positive class index number')
parser.add_argument('--positiveRate', type=float, default=0.33,
                    help='rate of positive sample')
# use Fedmatch dataloader
parser.add_argument('--task', type=str, default='lc-biid-c10')
parser.add_argument('--useFedmatchDataLoader', action='store_true', 
                    help='the same positive class index number')
parser.add_argument('--method', type=str, default='FedAvg')

# FL aggregator
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--communication_rounds', type=int, default=5000)
parser.add_argument('--classes_per_client', type=int, default=5)
parser.add_argument('--clientSelect_Rate', type=float, default=0.5)
# FedProx parameters
parser.add_argument('--mu', type=float, default=0.0)
parser.add_argument('--percentage', type=float, default=0.0)

opt, _ = parser.parse_known_args()


FedAVG_model_path = '/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/linxinyang/codebase/experiment/cache/model/local_model'
FedAVG_aggregated_model_path = '/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/linxinyang/codebase/experiment/cache/model/FedAVG_model.pth'
