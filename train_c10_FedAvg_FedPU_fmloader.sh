# # CIFAR10 FedAvg_PU_PUloss FedMatchloader-iid
python main.py   --dataset 'CIFAR10' --method 'FedAvg' --task 'FM-biib-c10' --use_PULoss --useFedmatchDataLoader --local_epochs 2 --communication_rounds 500 --pu_lr 0.01 --clientSelect_Rate 0.1


# CIFAR10 FedAvg_PU_PUloss FedMatchloader-non-iid
python main.py  --dataset 'CIFAR10' --method 'FedAvg' --task 'FM-bimb-c10' --use_PULoss --useFedmatchDataLoader --local_epochs 2 --communication_rounds 800 --pu_lr 0.01 --clientSelect_Rate 0.1
