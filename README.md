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
We include a dataset of environments and activities that agents have to perform in them. During the **Watch** phase and the training of the **Help** phase, we use a dataset of 5 environments. When evaluating the **Help** phase, we use a dataset of 2 held out environments.

The **Watch** phase consists of a set of episodes in 5 environments showing Alice performing the task. These episodes were generated using a planner, and they can be downloaded [here](). The training and testing splits for this phase can be found in [datasets/watch_scenes_split.json](datasets/watch_scenes_split.json). 

The **Help** phase, contains a set of environments and tasks definitions. You can find the *train* and *test* datasets used in `dataset/train_env_set_help.pik` and `dataset/test_env_set_help.pik`. Note that the *train* environments are independent, whereas the *test* environments match the tasks in the **Watch** test split.


### Create your own dataset 
You can also create your dataset, and modify it to incorporate new tasks. For that, run

```
python gen_data/vh_init.py --num-per-apartment {NUM_APT} --task {TASK_NAME}
```
Where `NUM_APT` corresponds to the number of episodes you want for each apartment and task and `TASK_NAME` corresponds to the task name you want to generate, which can be `setup_table`, `clean_table`, `put_fridge`, `prepare_food`, `read_book`, `watch_tv` or `all` to generate all the tasks.

After creating your dataset, you can create the data for the **Watch** phase running the *Alice alone* baseline (see [Evaluate Baselines](###Evaluate Baselines)).

You can then generate a dataset of tasks in a new environment where the tasks match those of the **Watch phase**. We do that in our work to make sure that the environment in the **Watch** phase is different than that in the **Help Phase** while having the same task specification. You can do that by running:



It will use the tasks from the test split of the **Watch** phase to create a **Help** dataset.



## Watch
Include here the code for the goal inference part

## Help
We provide planning and learning-based agents for the Helping stage. The agents have partial observability in the environment, and plan according to a belief that updates with new observations.

![](assets/collab_fig.gif)

### Train baselines


### Evaluate baselines
Below is the code to evaluate the different planning-based models.

```bash
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
