

# PU
#python main.py  --positiveRate 0.33 --P_Index_accordance --local_epochs 2 --randomIndex_num 2 --num_clients 20 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1

# baseline 1
#python main.py  --positiveRate 0.33 --positiveIndex '0' --P_Index_accordance --local_epochs 2 --randomIndex_num 2 --num_clients 20 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1

# baseline 2
python main.py  --positiveRate 0.5 --positiveIndex '0' --P_Index_accordance --local_epochs 2 --randomIndex_num 9 --num_clients 20 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1


