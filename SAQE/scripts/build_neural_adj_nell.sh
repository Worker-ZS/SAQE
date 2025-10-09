DATADIR=../../KG_data

if [ ! -d "./results" ]; then
  mkdir ./results
fi

if [ ! -d "/mnt/shared-storage-user/caozongsheng/neural_adj/" ]; then
  mkdir ./neural_adj
fi

# if [ ! -d "./neural_adj" ]; then
#   mkdir ./neural_adj
#CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/FB15k-betae --kbc_path kbc/FB15K/best_valid.model --fraction 10 --thrshd 0.0006 --num_scale 2.0 --neg_scale 6 --tasks 1p.2p.3p

#CUDA_VISIBLE_DEVICES=1 python main.py --data_path ${DATADIR}/FB15k-237-betae --kbc_path kbc/FB15K-237/best_valid.model --fraction 10 --thrshd 0.0002 --num_scale 1.5 --neg_scale 3 --tasks 1p.2p.3p

CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/NELL-betae --kbc_path kbc/NELL995/best_valid.model --fraction 10 --thrshd 0.0002 --num_scale 2.5 --neg_scale 6 --tasks 1p.2p.3p


cd ..
#1. 验证 membership function
bash scripts/valid_memb_func.sh QTO NELL 0 bspline

# 2. 网格验证
bash scripts/grid_valid.sh QTO NELL 0 bspline

# 3. 完整评估并重定向日志
bash scripts/eval_fs_nell-QTO.sh > log/nell-qto.log