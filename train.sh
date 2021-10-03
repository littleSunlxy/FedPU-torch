# debug Fedmatch dataloader

python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader  --task 'FedAVG-SL-iid' --num_clients 100 --local_epochs 10 --communication_rounds 400 --pu_lr 0.01 --clientSelect_Rate 0.05