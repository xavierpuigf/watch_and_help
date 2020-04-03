import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.unity_environment import UnityEnvironment
import pdb
import pickle
from algos.arena_mp2 import ArenaMP
import random
from agents import MCTS_agent, RL_agent
from arguments import get_args
from algos.arena import Arena
from algos.a2c_mp import A2C
from utils import utils_goals, utils_rl_agent
import ray
import atexit

if __name__ == '__main__':
    #ray.init(local_mode=True)

    ray.init()
    args = get_args()
    #args.task = 'setup_table'
    #args.num_per_apartment = '50'
    args.mode = 'full'
    num_agents = 1
    args.dataset_path = 'initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode)
    print(args.dataset_path)
    data = pickle.load(open(args.dataset_path, 'rb'))
    executable_args = {
            'file_name': args.executable_file,
            'x_display': 0,
            'no_graphics': True
    }

    with open(args.dataset_path, 'rb') as f:
        env_task_set = pickle.load(f)

    if args.debug:
        env_task_set = env_task_set[0]
        single_goal = [x for x,y in env_task_set['task_goal'][0].items() if y > 0 and x.split('_')[0] in ['on', 'inside']][0]

        if args.obs_type == 'mcts':
            env_task_set['init_rooms'] = ['kitchen']
        env_task_set['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        env_task_set = [env_task_set]

    else:
        if args.task_set != 'full':
            env_task_set = [env_task for env_task in env_task_set if env_task['task_name'] == args.task_set]


    env_task_set = [env for env in env_task_set if env['env_id'] == 0]
    print('Number of episides: {}'.format(len(env_task_set)))

    agent_goal = 'grab'
    if args.task_type == 'put':
        agent_goal = 'put'



    # args_mcts = dict(unity_env=env,
    #                    recursive=True,
    #                    max_episode_length=5,
    #                    num_simulation=100,
    #                    max_rollout_steps=3,
    #                    c_init=0.1,
    #                    c_base=1000000,
    #                    num_samples=1,
    #                    num_processes=1,
    #                    logging=True)


    graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args.max_num_objects,
                                              max_num_edges=args.max_num_edges, current_task=None, simulator_type='unity')

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent2 = {'agent_id': 1, 'char_index': 0,
                   'args': args, 'graph_helper': graph_helper}


    def RL_agent_fn():
        return RL_agent(**args_agent2)

    def env_fn(env_id):
        return UnityEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                agent_goals=[agent_goal],
                                observation_types=[args.obs_type],
                                use_editor=args.use_editor,
                                executable_args=executable_args,
                                base_port=args.base_port,
                                seed=env_id)

    # args_agent1.update(args_mcts)


    #args_agent2.update(args_common)
    #agents = [MCTS_agent(**args_agent1), RL_agent(**args_agent2)]
    #agents = [RL_agent(**args_agent2)]

    agents = [RL_agent_fn]

    arenas = [ArenaMP.remote(arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
    #arenas = [ArenaMP(arena_id, env_fn, agents) for arena_id in range(args.num_processes)]

    a2c = A2C(arenas, graph_helper, args)

    a2c.train()
    pdb.set_trace()
    # try:
    #     a2c = A2C(arenas, graph_helper, args)
    # except:
    #     for arena in arenas:
    #         del(arena)
    #     pdb.set_trace()
    #ray.shutdown()
