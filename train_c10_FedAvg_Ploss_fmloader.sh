# CIFAR10 FedAvg_PU_Ploss FedMatchloader-iid
python main.py   --dataset 'CIFAR10' --method 'FedAvg' --task 'FM-biib-c10' --useFedmatchDataLoader --local_epochs 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1


# # CIFAR10 FedAvg_PU_Ploss FedMatchloader-non-iid
python main.py   --dataset 'CIFAR10' --method 'FedAvg' --task 'FM-bimb-c10' --useFedmatchDataLoader --local_epochs 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
