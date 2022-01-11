# train FedProx

# exp in original paper
#01.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.01 --percentage 0.0 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
#02.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.01 --percentage 0.5 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
##03.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.01 --percentage 0.9 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1

#04.log
python main.py   --dataset 'MNIST' --method 'FedProx' --mu 0.00 --percentage 0.0 --num_clients 100 --pu_batchsize 64 --classes_per_client 10 --P_Index_accordance --positiveRate 0.99 --randomIndex_num 10 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
#05.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.01 --percentage 0.5 --num_clients 100 --pu_batchsize 128 --classes_per_client 10 --P_Index_accordance --positiveRate 0.99 --randomIndex_num 10 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
#06.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --mu 0.01 --percentage 0.9 --num_clients 100 --pu_batchsize 128 --classes_per_client 10 --P_Index_accordance --positiveRate 0.99 --randomIndex_num 10 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1

#04.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --usePU --mu 0.01 --percentage 0.0 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
##05.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --usePU --mu 0.01 --percentage 0.5 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
##06.log
#python main.py   --dataset 'CIFAR10' --method 'FedProx' --usePU --mu 0.01 --percentage 0.9 --num_clients 100 --pu_batchsize 512 --classes_per_client 2 --P_Index_accordance --positiveRate 0.33 --randomIndex_num 2 --communication_rounds 200 --pu_lr 0.01 --clientSelect_Rate 0.1
