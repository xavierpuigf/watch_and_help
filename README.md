# Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration




This is the official implementation of the paper *Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration*. In this work, we introduce Watch-And-Help (WAH), a challenge for testing social intelligence in agents. In WAH, an AI agent needs to help a human-like agent perform a complex household task efficiently. To succeed, the AI agent needs to i) understand the underlying goal of the task by watching a single demonstration of the human-like agent performing the same task (social perception), and ii) coordinate with the human-like agent to solve the task in an unseen environment as fast as possible (human-AI collaboration).

![](assets/cover_fig_final.png)

We provide a dataset of tasks to evaluate the challenge, as well as different baselines consisting on learning and planning-based agents.

## Setup
### Get the VirtualHome Simulator and API
Clone the VirtualHome API repository one folder above this repository

```bash
cd ..
git clone https://github.com/xavierpuigf/virtualhome.git
```

Download the simulator

- [Download](http://virtual-home.org/release/simulator/linux_sim.zip) Linux x86-64 version.
- [Download](http://virtual-home.org/release/simulator/mac_sim.zip) Mac OS X version.
- [Download](http://virtual-home.org/release/simulator/windows_sim.zip) Windows version.

### Install Requirements





## Dataset

## Watch
Include here the code for the goal inference part

## Help
We provide planning and learning-based agents for the Helping stage. The agents have partial observability on the environment, and plan according to a belief that updates with new observations.

### Train baselines


### Evaluate baselines
Below is the code to evaluate the different planning-based models.

```
# Alice alone
python testing_agents/test_single_agent.py

# Bob planner true goal
python testing_agents/test_hp.py

# Bob planner predicted goal
python testing_agents/test_hp_pred_goal.py

# Bob planner random goal
python testing_agents/test_hp_random_goal.py

# Bob random actions
python testing_agents/test_hp_random_action.py
```

Below is the code to evaluate the learning-based methods

```
CUDA_VISIBLE_DEVICES=0 python evaluate_a2c.py \
--max-num-edges 10 --max-episode-length 250 --obs_type partial \
--nb_episodes 100000 --save-interval 200 \
--base_net TF --log-interval 1 --long-log 50 --num-processes 1 \
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
