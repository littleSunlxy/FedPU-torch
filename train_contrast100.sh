# client 100 contrast
CUDA_VISIBLE_DEVICES=4 python main.py  --dataset 'CIFAR10' --pu_batchsize 128 --positiveRate 0.1 --pu_lr 0.01 --local_epochs 1 --P_Index_accordance --randomIndex_num 10 --num_clients 100 --communication_rounds 400 --classes_per_client 10 --clientSelect_Rate 0.05 > nohup.out 2>&2 &
