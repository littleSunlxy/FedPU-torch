# client 100 contrast
python main.py  --dataset 'CIFAR10' --pu_batchsize 10 --positiveRate 0.99 --pu_lr 0.01 --local_epochs 1 --P_Index_accordance --randomIndex_num 2 --num_clients 100 --communication_rounds 400 --classes_per_client 2 --clientSelect_Rate 0.05
