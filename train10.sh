# PU
python main.py  --positiveRate 0.33 --P_Index_accordance --local_epochs 5 --randomIndex_num 5 --num_clients 10 --communication_rounds 500 --classes_per_client 10 --clientSelect_Rate 0.2

# baseline 1
python main.py  --positiveRate 0.33 --positiveIndex '0' --P_Index_accordance --local_epochs 5 --randomIndex_num 5 --num_clients 10 --communication_rounds 500 --classes_per_client 10 --clientSelect_Rate 0.2

# baseline 2
python main.py  --positiveRate 0.33 --P_Index_accordance --local_epochs 5 --randomIndex_num 5 --num_clients 10 --communication_rounds 500 --classes_per_client 10 --clientSelect_Rate 0.2


