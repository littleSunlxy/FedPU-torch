
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pu_lr', type=float, default=0.1)
parser.add_argument('--pu_weight_decay', type=float, default=5e-3)
parser.add_argument('--pu_batchsize', type=int, default=1024)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dataset', type=str, default='MNIST')
#parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--num_classes', type=int, default=10)
#parser.add_argument('--label_dir', type=str, default='/workdir/linxinyang/experiment/data/cifar10/')
parser.add_argument('--label_dir', type=str, default='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/linxinyang/codebase/experiment/data/mnist/')

# pu on clients
parser.add_argument('--pu_weight', type=float, default=1)#1
parser.add_argument('--local_epochs', type=int, default=2)
#parser.add_argument('--positiveIndex', type=str, default='k')  # 第k类为负类
#parser.add_argument('--positiveIndex', type=str, default='0') #仅用ploss
parser.add_argument('--positiveIndex', type=str, default='randomIndexList') #随机选两个标签为负类
parser.add_argument('--P_Index_accordance', action='store_true') #随机选两个标签为负类

parser.add_argument('--positiveRate', type=float, default=0.33) #1
parser.add_argument('--randomIndex_num', type=int, default=2)

# FL aggregator
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--communication_rounds', type=int, default=5000)
parser.add_argument('--classes_per_client', type=int, default=5)
#parser.add_argument('--participating_ratio', type=float, default=0.7)
parser.add_argument('--clientSelect_Rate', type=float, default=0.5)
parser.add_argument('--log_name', type=str, default='out.log')
parser.add_argument('--imagename', type=str, default='5.23.6.jpg')

opt, _ = parser.parse_known_args()


FedAVG_model_path = '/workdir/linxinyang/experiment/cache/model/local_model'
FedAVG_aggregated_model_path = '/workdir/linxinyang/experiment/cache/model/FedAVG_model.pth'

# FedAVG_model_path = '/home/lxx-006/lxy/fmpu/cache/model/local_model'
# FedAVG_aggregated_model_path = '/home/lxx-006/lxy/fmpu/cache/model/FedAVG_model.pth'