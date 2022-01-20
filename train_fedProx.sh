# train FedProx

# train FedPU FedMatch-iid
#python main.py  --useFedmatchDataLoader  --dataset 'CIFAR10' --method 'FedPU' --usePU --local_epochs 10 --num_clients 100 --pu_batchsize 512 --communication_rounds 500 --pu_lr 0.01 --clientSelect_Rate 0.05

# train FedProx FedMatch-iid
#python main.py  --useFedmatchDataLoader  --dataset 'CIFAR10' --method 'FedProx' --usePU --local_epochs 20 --pu_weight 0.3 --mu 0.1 --percentage 0.0 --num_clients 100 --pu_batchsize 512 --communication_rounds 500 --pu_lr 0.01 --clientSelect_Rate 0.05


# train FedPU FedMatch-noniid
#python main.py  --useFedmatchDataLoader  --dataset 'CIFAR10' --method 'FedPU' --local_epochs 20 --mu 0.10 --percentage 0.0 --num_clients 100 --pu_batchsize 512 --communication_rounds 500 --pu_lr 0.01 --clientSelect_Rate 0.05

# train FedProx FedMatch-noniid
#python main.py  --useFedmatchDataLoader  --dataset 'CIFAR10' --task 'lc-bimb-c10' --method 'FedProx' --usePU --local_epochs 20 --mu 0.10 --pu_weight 1.0 --percentage 0.0 --num_clients 100 --pu_batchsize 512 --communication_rounds 500 --pu_lr 0.01 --clientSelect_Rate 0.05


# use my data spilt setting
python main.py   --dataset 'CIFAR10' --method 'FedProx' --usePU --adjust_lr --local_epochs 10 --communication_rounds 400 --pu_weight 0 --mu 0.10 --num_clients 100 --pu_batchsize 512 --classes_per_client 5 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --pu_lr 0.01 --clientSelect_Rate 0.2

