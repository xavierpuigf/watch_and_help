CUDA_VISIBLE_DEVICES=0 python test_a2c.py \
--num-per-apartment 3 \
--debug --max-num-edges 500 --max-episode-length 30 \
--balanced_sample --neg_ratio 0.5 --batch_size 16 \
--obs_type full --gamma 0.95 --lr 1e-4 \
--logging