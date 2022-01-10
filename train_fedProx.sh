# train FedProx

# exp in original paper
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.0 --percentage 0.0 --num_clients 100 --pu_batchsize 2048 --classes_per_client 10 --P_Index_accordance --positiveRate 0.99 --randomIndex_num 10 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1

# exp for FedPU
python main.py   --dataset 'CIFAR10' --method 'FedProx' --usePU --mu 0.0 --percentage 0.0 --num_clients 100 --pu_batchsize 2048 --classes_per_client 10 --P_Index_accordance --positiveRate 0.5 --randomIndex_num 10 --communication_rounds 50 --pu_lr 0.01 --clientSelect_Rate 0.1