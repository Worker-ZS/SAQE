DATADIR=../KG_data
MODELDIR=../SAQE-M/models
#ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/FB15k-237_15_0.00015_2.0.pt  
ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/FB15k-237_10_0.0002_1.5.pt  
BACKBONE=SAQE

CUDA_VISIBLE_DEVICES=0 python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --adj_path ${ADJ_PATH} --dataname FB15k-237 --backbone_type ${BACKBONE}