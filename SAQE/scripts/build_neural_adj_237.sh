DATADIR=../../KG_data

if [ ! -d "./results" ]; then
  mkdir ./results
fi

if [ ! -d "/mnt/shared-storage-user/caozongsheng/neural_adj/" ]; then
  mkdir /mnt/shared-storage-user/caozongsheng/neural_adj
fi


CUDA_VISIBLE_DEVICES=0 python main.py \
  --data_path ${DATADIR}/FB15k-237-betae \
  --kbc_path kbc/FB15K-237/best_valid.model \
  --fraction 10 \
  --thrshd 0.0002 \
  --num_scale 1.5 \
  --neg_scale 3 \
  --tasks 1p.2p.3p

cd ..
# #1. 验证 membership function
# bash scripts/valid_memb_func.sh QTO FB15k-237 0 bspline
# bash scripts/valid_memb_func.sh SAQE FB15k-237 0 bspline

# # 2. 网格验证
# bash scripts/grid_valid.sh QTO FB15k-237 0 bspline
#bash scripts/grid_valid.sh SAQE FB15k-237 0 bspline

# # 3. 完整评估并重定向日志
# bash scripts/eval_fs_fb15k-237-QTO.sh > log/fb15k-237-qto.log
#bash scripts/eval_fs_fb15k-237-SAQE.sh > log/fb15k-237-saqe.log
# echo "全部步骤执行完毕"

#CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/NELL-betae --kbc_path kbc/NELL995/best_valid.model --fraction 10 --thrshd 0.0002 --num_scale 2.5 --neg_scale 6 --tasks 1p.2p.3p