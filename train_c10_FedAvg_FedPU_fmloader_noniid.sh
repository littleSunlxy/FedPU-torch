# CIFAR10 FedAvg_PU_PUloss FedMatchloader-non-iid
python main.py  --dataset 'CIFAR10' --method 'FedAvg' --task 'FM-bimb-c10' --useFedmatchDataLoader --local_epochs 2 --communication_rounds 800 --pu_lr 0.01 --clientSelect_Rate 0.1
