# vh_multiagent_models
Code for models exploring multiagents in VirtualHome


Run training with python trainer.py

## Evaluate Models
Hybrid model

```
CUDA_VISIBLE_DEVICES=4 python evaluate_a2c.py \
--num-per-apartment 3 --max-num-edges 10 --max-episode-length 250 --batch_size 32 --obs_type mcts \
--gamma 0.95 --lr 1e-4 --task_type find  --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --base-port 8589 --num-processes 1 \
--agent_type hrl_mcts --num_steps_mcts 40 --use-alice \
--load-model trained_models/env.virtualhome/\
task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/\
mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32_finetuned/\
stepmcts.50-lep.250-teleport.False-gtgraph-forcepred/2000.pt

```

```
model_path_lowlevel = ('/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/tshu/vh_multiagent_models/'
            'trained_models/env.virtualhome/task.put-numproc.1-obstype.mcts-sim.unity/'\
            'taskset.full/mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.1.0-cgoal.0.0-lr0.0001/26200_all.pt')
```