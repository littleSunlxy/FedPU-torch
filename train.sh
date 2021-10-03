# debug Fedmatch dataloader

python main.py   --dataset 'CIFAR10'  --useFedmatchDataLoader --positiveRate 0.01 --P_Index_accordance --randomIndex_num 2 --num_clients 100 --communication_rounds 200 --classes_per_client 10 --clientSelect_Rate 0.1