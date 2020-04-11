import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.python_environment import PythonEnvironment
from envs.unity_environment import UnityEnvironment
import pdb
import pickle
import random
from agents import MCTS_agent, RL_agent
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

        #env_task_set['init_rooms'] = ['kitchen']

        env_task_set['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        env_task_set = [env_task_set]
    else:
        if args.task_set != 'full':
            env_task_set = [env_task for env_task in env_task_set if env_task['task_name'] == args.task_set]



    print('Number of episides: {}'.format(len(env_task_set)))

    agent_goal = 'grab'
    if args.task_type == 'put':
        agent_goal = 'put'

    def env_fn(env_id):
        if args.simulator_type == 'unity':
            return UnityEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                                    port_id=env_id,
                                    env_task_set=env_task_set,
                                    agent_goals=[agent_goal],
                                    observation_types=[args.obs_type],
                                    use_editor=args.use_editor,
                                    executable_args=executable_args,
                                    base_port=args.base_port)
        else:
            return PythonEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                                    env_task_set=env_task_set,
                                    agent_goals=[agent_goal],
                                    observation_types=[args.obs_type],
                                    seed=env_id)


    graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args.max_num_objects,
                                              max_num_edges=args.max_num_edges, current_task=None,
                                              simulator_type=args.simulator_type)

    def RL_agent_fn(arena_id, env):
        args_agent2 = {'agent_id': 1, 'char_index': 0,
                       'args': args, 'graph_helper': graph_helper}
        args_agent2['seed'] = arena_id
        return RL_agent(**args_agent2)

    agents = [RL_agent_fn]
    if args.num_processes > 1:
        ArenaMP = ray.remote(ArenaMP) #, max_reconstructions=ray.ray_constants.INFINITE_RECONSTRUCTION)
        arenas = [ArenaMP.remote(arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C_MP(arenas, graph_helper, args)
    else:
        arenas = [ArenaMP(arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C(arenas, graph_helper, args)
    a2c.train()
    pdb.set_trace()
