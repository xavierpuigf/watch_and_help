python src/predicate-train.py \
--gpu_id 0 \
--model_lr_rate 3e-4 \
--batch_size 8 \
--demo_hidden 512 \
--model_type lstmavg \
--inputtype graphinput \
--dropout 0 \
--single 1 \
--checkpoints checkpoints/test
