# PU
#python main.py  --positiveRate 0.33  --local_epochs 3 --P_Index_accordance --randomIndex_num 2 --num_clients 50 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1

# baseline 1
python main.py  --positiveRate 0.33  --positiveIndex '0' --local_epochs 3 --P_Index_accordance --randomIndex_num 2 --num_clients 50 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1

# baseline 2
#python main.py  --positiveRate 0.99  --positiveIndex '0' --local_epochs 3 --P_Index_accordance --randomIndex_num 9 --num_clients 50 --communication_rounds 2000 --classes_per_client 10 --clientSelect_Rate 0.1


