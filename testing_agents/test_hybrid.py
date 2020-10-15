import sys
import os
import ipdb
import pickle
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, HRL_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from algos.a2c_mp import A2C as A2C_MP
from utils import utils_goals, utils_rl_agent
import ray

if __name__ == '__main__':
    args = get_args()
    
    args.max_episode_length = 250
    args.num_per_apartment = 20
    args.mode = 'hybrid_truegoal'
    args.evaluation = True
    args.use_alice = True
    args.obs_type = 'partial'
    args.dataset_path = './dataset/test_env_set_help.pik'

    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    args.record_dir = '../test_results/multiBob_env_task_set_{}_{}'.format(args.num_per_apartment, args.mode)
    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    
    if args.task_set != 'full':
        env_task_set = [env_task for env_task in env_task_set if env_task['task_name'] == args.task_set]




    agent_goal = 'full'
    args.task_type = 'full'
    num_agents = 1
    agent_goals = [agent_goal]
    if args.use_alice:
        num_agents += 1
        observation_types = ['partial', args.obs_type]
        agent_goals.append(agent_goal)
        rl_agent_id = 2
    else:
        rl_agent_id = 1
        observation_types = [args.obs_type]

    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    num_tries = 5
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]

    def env_fn(env_id):
        return UnityEnvironment(num_agents=num_agents,
                                max_episode_length=args.max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                agent_goals=agent_goals,
                                observation_types=observation_types,
                                use_editor=args.use_editor,
                                executable_args=executable_args,
                                base_port=args.base_port,
                                seed=None)


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
                           logging=True,
                           logging_graphs=True)

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
    
    a2c.load_model(args.load_model)

    test_results = {}
    for iter_id in range(num_tries):
        seed = iter_id

        for episode_id in tqdm(range(len(episode_ids))):
            print(episode_id, len(episode_ids))
            log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(env_task_set[episode_id]['task_id'],
                                                                                env_task_set[episode_id]['task_name'],
                                                                                seed)
            if os.path.isfile(log_file_name):
                print('exsits')
                continue
            if True:
                for agent in arenas[0].agents:
                    agent.seed = seed
                res = a2c.eval(episode_id)
                finished = res[1][0]['finished']
                length = len(res[1][0]['action'][0])
                info_results = {
                    'finished': finished,
                    'L': length,
                    'task_id': arenas[0].env.task_id,
                    'env_id': arenas[0].env.env_id,
                    'task_name': arenas[0].env.task_name,
                    'gt_goals': arenas[0].env.task_goal[0],
                    'goals_finished': res[1][0]['goals_finished'],
                    'goals': arenas[0].env.task_goal,
                    'obs': res[1][0]['obs'],
                    'action': res[1][0]['action']
                }
                
                S[episode_id].append(finished)
                L[episode_id].append(length)
                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                with open(log_file_name, 'wb') as flog:
                    pickle.dump(info_results, flog)
            else:
                arenas[0].reset_env()
        test_results[episode_id] = {'S': S[episode_id], 'L': L[episode_id]}
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
