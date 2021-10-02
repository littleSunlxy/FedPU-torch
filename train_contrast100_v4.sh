# client 100 contrast
CUDA_VISIBLE_DEVICES=3 python main.py  --dataset 'CIFAR10' --pu_batchsize 1024 --positiveRate 0.33 --pu_lr 0.01 --local_epochs 20 --P_Index_accordance --randomIndex_num 5 --num_clients 100 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.05 > nohup_v4.out 2>&2 &
