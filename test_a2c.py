import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.python_environment import PythonEnvironment
from envs.unity_environment import UnityEnvironment
import pdb
import pickle
import random
import copy
from agents import MCTS_agent, RL_agent, HRL_agent
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
    #args.dataset_path = 'initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode)
    args.dataset_path = 'initial_environments/data/init_envs/test_env_set_help_20_neurips.pik'

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
        # # debug 1: 1 predicate, 1 room
        # env_task_set = env_task_set[0]
        # single_goal = [x for x,y in env_task_set['task_goal'][0].items() if y > 0 and x.split('_')[0] in ['on', 'inside']][0]

        # if args.obs_type == 'mcts':
        #     env_task_set['init_rooms'] = ['kitchen']
        # env_task_set['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        # env_task_set = [env_task_set]

        # debug 2: multiple predicates, 1 room
        env_task_set0 = copy.deepcopy(env_task_set)
        env_task_set = []
        env_task = env_task_set0[0]
        single_goals = [x for x,y in env_task_set0[0]['task_goal'][0].items() if y > 0 and x.split('_')[0] in ['on', 'inside']]
        # pdb.set_trace()
        if args.obs_type == 'mcts':
            env_task['init_rooms'] = ['kitchen']

        for single_goal in single_goals:
            env_task_new = copy.deepcopy(env_task)
            env_task_new['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
            env_task_set.append(env_task_new)

        print('# env_task for debug:', len(env_task_set))

        for env_task in env_task_set:
            print(env_task['task_name'], env_task['task_goal'][0])

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
            return UnityEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                                    port_id=env_id,
                                    env_task_set=env_task_set,
                                    agent_goals=agent_goals,
                                    observation_types=observation_types,
                                    use_editor=args.use_editor,
                                    executable_args=executable_args,
                                    base_port=args.base_port)
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

    def RL_agent_fn(arena_id, env):
        args_agent2 = {'agent_id': 1, 'char_index': 0,
                       'args': args, 'graph_helper': graph_helper}
        args_agent2['seed'] = arena_id
        return RL_agent(**args_agent2)


    def HRL_agent_fn(arena_id, env):
        args_agent2 = {'agent_id': rl_agent_id, 'char_index': rl_agent_id - 1,
                       'args': args, 'graph_helper': graph_helper}
        args_agent2['seed'] = arena_id
        return HRL_agent(**args_agent2)




    # agents = [RL_agent_fn]
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
