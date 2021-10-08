# debug Fedmatch dataloader

## iid  FedAVG-SL
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-biid-c10' --method 'FedAVG-SL' --num_clients 100 --local_epochs 1 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05
#
## iid  FedPU
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-biid-c10' --method 'FedPU' --num_clients 100 --local_epochs 1 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05

# non-iid FedAVG-SL
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-bimb-c10' --method 'FedAVG-SL' --num_clients 100 --local_epochs 5 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05
#
### non-iid  FedPU
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-bimb-c10' --method 'FedPU' --num_clients 100 --local_epochs 1 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05


python main.py   --dataset 'CIFAR10' --method 'FedPU-SL' --num_clients 1 --pu_batchsize 2048 --classes_per_client 10 --P_Index_accordance --positiveRate 0.5 --randomIndex_num 10 --local_epochs 5 --communication_rounds 50 --pu_lr 0.01 --clientSelect_Rate 1.0
