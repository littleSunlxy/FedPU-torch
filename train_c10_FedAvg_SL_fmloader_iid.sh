# CIFAR10 FedAvg_SL FedMatchloader-iid
python main.py  --dataset 'CIFAR10' --method 'FedAvg_SL' --task 'FM-biib-c10' --useFedmatchDataLoader --local_epochs 2 --communication_rounds 2000 --pu_lr 0.01 --clientSelect_Rate 0.1
