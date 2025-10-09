DATADIR=../KG_data
MODELDIR=../SAQE-M/models

BACKBONE=$1
DATASET=$2
CUDA_ID=$3


if [ ${DATASET} == 'NELL' ]; then
    ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/NELL_10_0.0002_2.5.pt
    DATASET_L=nell
fi

if [ ${DATASET} == 'FB15k' ]; then
    #ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/FB15k_10_0.0006_2.0.pt
    ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/FB15k_10_0.01_2.0.pt
    DATASET_L=fb15k
fi

if [ ${DATASET} == 'FB15k-237' ]; then
    ADJ_PATH=/mnt/shared-storage-user/zz/neural_adj/FB15k-237_10_0.0002_1.5.pt  
    DATASET_L=fb15k-237
fi 

FILE=./params/${DATASET}_${BACKBONE}_1p_4.7_validation.txt
if [ ! -f "$FILE" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 main.py --do_valid --do_save --data_path ${DATADIR}/${DATASET}-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/${DATASET_L}-betae --cqd discrete --cuda --adj_path ${ADJ_PATH} --dataname ${DATASET} --backbone_type ${BACKBONE}
fi



