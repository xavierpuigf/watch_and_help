python watch/predicate-train.py \
--gpu_id 1 \
--batch_size 4 \
--demo_hidden 512 \
--model_type lstmavg \
--dropout 0 \
--inputtype graphinput \
--inference 1 \
--single 1 \
--resume 'checkpoints/demo2predicate-best_model.ckpt' \
--checkpoint checkpoints/test
