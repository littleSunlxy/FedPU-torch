# debug Fedmatch dataloader

## iid  FedAVG-SL
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-biid-c10' --method 'FedAVG-SL' --num_clients 100 --local_epochs 1 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05
#
## iid  FedAVG-FedPU
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-biid-c10' --method 'FedPU' --num_clients 100 --local_epochs 1 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05

# non-iid
python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-bimb-c10' --method 'FedAVG-SL' --num_clients 100 --local_epochs 5 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05
#
## non-iid  FedAVG-FedPU
#python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'lc-bimb-c10' --method 'FedPU' --num_clients 100 --local_epochs 10 --communication_rounds 250 --pu_lr 0.01 --clientSelect_Rate 0.05
