"""
CUDA_VISIBLE_DEVICES=0 python training_agents/test_a2c.py --num-per-apartment 3 --max-num-edges 10 \
--max-episode-length 250 --batch_size 32 --obs_type partial --gamma 0.95 \
--lr 1e-4 --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --logging --base-port 8681 \
--num-processes 5 --teleport --executable_file ../../executable/linux_exec_v2.x86_64 \
--agent_type hrl_mcts --num_steps_mcts 24


NO TELEPORT
CUDA_VISIBLE_DEVICES=0 python test_a2c.py --num-per-apartment 3 --max-num-edges 10 \
--max-episode-length 250 --batch_size 32 --obs_type mcts --gamma 0.95 --lr 1e-4 \
--task_type find  --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --logging --base-port 8681 --num-processes 5 --teleports \
--executable_file /data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.05.29.x86_64 \
--agent_type hrl_mcts --num_steps_mcts 24



CUDA_VISIBLE_DEVICES=5 python test_a2c.py --num-per-apartment 3 --max-num-edges 10 \
--max-episode-length 250 --batch_size 32 --obs_type mcts --gamma 0.95 --lr 1e-4 \
--task_type find  --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --logging --base-port 8681 --num-processes 5 \
--agent_type hrl_mcts --num_steps_mcts 50 \
--load-model trained_models/env.virtualhome/\
task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/\
mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32/\
stepmcts.24-lep.250-teleport.True-gtgraph/4200.pt


# WITH ALICE
CUDA_VISIBLE_DEVICES=6 python test_a2c.py --num-per-apartment 3 --max-num-edges 10 \
--max-episode-length 250 --batch_size 32 --obs_type mcts --gamma 0.95 --lr 1e-4 \
--task_type find  --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --logging --base-port 8781 --num-processes 5 \
--agent_type hrl_mcts --num_steps_mcts 20 --use-alice --max-number-steps 50 \
--load-model trained_models/env.virtualhome/\
task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/\
mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32_finetuned/\
stepmcts.50-lep.250-teleport.False-gtgraph-forcepred/2000.pt

"""
import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}/..')

from envs.unity_environment import UnityEnvironment
import pdb
import pickle
import random
import copy
from agents import MCTS_agent, HRL_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from algos.a2c import A2C
from algos.a2c_mp import A2C as A2C_MP
from utils import utils_goals, utils_rl_agent
import ray

if __name__ == '__main__':
    args = get_args()
    #args.task = 'setup_table'
    #args.num_per_apartment = '50'
    if args.num_processes > 1:
        ray.init()

    args.mode = 'full'
    num_agents = 1
    args.dataset_path = 'dataset/train_env_set_help.pik'

    data = pickle.load(open(args.dataset_path, 'rb'))
    
    
    executable_args = {
            'file_name': args.executable_file,
            'x_display': 0,
            'no_graphics': True
    }

    with open(args.dataset_path, 'rb') as f:
        env_task_set = pickle.load(f)


    for env in env_task_set:
        g = env['init_graph']
        id2node = {node['id']: node['class_name'] for node in g['nodes']}
        cloth_ids = [node['id'] for node in g['nodes'] if node['class_name'] in ["clothespile"]]
        g['nodes'] = [node for node in g['nodes'] if node['id'] not in cloth_ids]
        g['edges'] = [edge for edge in g['edges'] if edge['from_id'] not in cloth_ids and edge['to_id'] not in cloth_ids]

    
    #env_task_set = [env_task_set[220]]
    if args.debug:
        # # debug 1: 1 predicate, 1 room
        # env_task_set = env_task_set[0]
        # single_goal = [x for x,y in env_task_set['task_goal'][0].items() if y > 0 and x.split('_')[0] in ['on', 'inside']][0]

        # if args.obs_type == 'mcts':
        #     env_task_set['init_rooms'] = ['kitchen']
        # env_task_set['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        # env_task_set = [env_task_set]

        # debug 2: multiple predicates, 1 room
        # env_task_set0 = copy.deepcopy(env_task_set)
        # env_task_set = []
        # env_task = env_task_set0[0]
        # single_goals = [x for x,y in env_task_set0[0]['task_goal'][0].items() if y > 0 and x.split('_')[0] in ['on', 'inside']]
        # # pdb.set_trace()
        # if args.obs_type == 'mcts':
        #     env_task['init_rooms'] = ['kitchen']
        #
        # for single_goal in single_goals:
        #     env_task_new = copy.deepcopy(env_task)
        #     env_task_new['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        #     env_task_set.append(env_task_new)
        #
        # print('# env_task for debug:', len(env_task_set))
        #
        # for env_task in env_task_set:
        #     print(env_task['task_name'], env_task['task_goal'][0])


        # One room, multi preds in same task
        env_task_set = [env_task_set[0]]

        # # debug 3: 1 predicate, multiple rooms
        # env_task_set0 = copy.deepcopy(env_task_set)
        # env_task_set = []
        # for env_task in env_task_set0:
        #   if env_task['task_name'] == 'setup_table':
        #     single_goal = [x for x, y in env_task['task_goal'][0].items() if y > 0 and x.split('_')[1] == 'plate']  
        #     if len(single_goal) == 1:
        #       env_task_new = copy.deepcopy(env_task)
        #       env_task_new['task_goal'] = {0: {single_goal[0]: 1}, 1: {single_goal[0]: 1}}
        #       env_task_set.append(env_task_new)
        # print('# env_task for debug:', len(env_task_set))
    else:
        if args.task_set != 'full':
            env_task_set = [env_task for env_task in env_task_set if env_task['task_name'] == args.task_set]


    # env_task_set = [[env for env in env_task_set if env['env_id'] == 2][0]]
    print('Number of episides: {}'.format(len(env_task_set)))

    agent_goal = 'full'
    args.task_type = 'full'
    # if args.task_type == 'put':
    #     agent_goal = 'put'

    agent_goals = [agent_goal]
    if args.use_alice:
        num_agents += 1
        observation_types = ['mcts', args.obs_type]
        agent_goals.append(agent_goal)
        rl_agent_id = 2
    else:
        rl_agent_id = 1
        observation_types = [args.obs_type]

    def env_fn(env_id):
        if args.simulator_type == 'unity':

            return UnityEnvironment(num_agents=num_agents, 
                                    max_episode_length=args.max_episode_length,
                                    port_id=env_id,
                                    env_task_set=env_task_set,
                                    agent_goals=agent_goals,
                                    observation_types=observation_types,
                                    use_editor=args.use_editor,
                                    executable_args=executable_args,
                                    base_port=args.base_port,
                                    seed=env_id)
        else:
            return PythonEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                                    env_task_set=env_task_set,
                                    agent_goals=agent_goals,
                                    observation_types=observation_types,
                                    seed=env_id)


    graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args.max_num_objects,
                                              max_num_edges=args.max_num_edges, current_task=None,
                                              simulator_type=args.simulator_type)


    def MCTS_agent_fn(arena_id, env):
        args_mcts = dict(recursive=False,
                           max_episode_length=5,
                           num_simulation=100,
                           max_rollout_steps=5,
                           c_init=0.1,
                           c_base=1000000,
                           num_samples=1,
                           num_processes=1,
                           logging=False,
                           logging_graphs=False)

        args_mcts['agent_id'] = 1
        args_mcts['char_index'] = 0
        return MCTS_agent(**args_mcts)



    def HRL_agent_fn(arena_id, env):
        args_agent2 = {'agent_id': rl_agent_id, 'char_index': rl_agent_id - 1,
                       'args': args, 'graph_helper': graph_helper}
        args_agent2['seed'] = arena_id
        return HRL_agent(**args_agent2)




    agents = [HRL_agent_fn]

    if args.use_alice:
        agents = [MCTS_agent_fn] + agents
    if args.num_processes > 1:
        ArenaMP = ray.remote(ArenaMP) #, max_reconstructions=ray.ray_constants.INFINITE_RECONSTRUCTION)
        arenas = [ArenaMP.remote(args.max_number_steps, arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C_MP(arenas, graph_helper, args)
    else:
        arenas = [ArenaMP(args.max_number_steps, arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C_MP(arenas, graph_helper, args)
    a2c.train()
    pdb.set_trace()
