
"""
CUDA_VISIBLE_DEVICES=4 python evaluate_a2c.py \
--num-per-apartment 3 --max-num-edges 10 --max-episode-length 250 --batch_size 32 --obs_type mcts \
--gamma 0.95 --lr 1e-4 --task_type find  --nb_episodes 100000 --save-interval 200 --simulator-type unity \
--base_net TF --log-interval 1 --long-log 50 --base-port 8589 --num-processes 1 \
--agent_type hrl_mcts --num_steps_mcts 40 --use-alice \
--load-model trained_models/env.virtualhome/\
task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/\
mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32_finetuned/\
stepmcts.50-lep.250-teleport.False-gtgraph-forcepred/2000.pt

"""
import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.python_environment import PythonEnvironment
from envs.unity_environment import UnityEnvironment
import pdb
from pathlib import Path
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
    # args.task = 'setup_table'
    # args.num_per_apartment = '50'
    # args.mode = 'full'
    # args.dataset_path = 'initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(args.task,
    #                                                                                        args.num_per_apartment,
    #                                                                                     args.mode)
    # data = pickle.load(open(args.dataset_path, 'rb'))
    args.max_episode_length = 250
    args.num_per_apartment = '20'
    args.base_port = 8082
    args.evaluation = True
    args.mode = 'check_neurips_RL_MCTS_multiple'
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.27.x86_64'

    # env_task_set = pickle.load(open('initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'rb'))
    # env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_help_20_neurips.pik', 'rb'))
    env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_help_10_multitask_neurips.pik', 'rb'))


    for env in env_task_set:
        if env['env_id'] == 6:
            g = env['init_graph']
            door_ids = [302, 213]
            g['nodes'] = [node for node in g['nodes'] if node['id'] not in door_ids]
            g['edges'] = [edge for edge in g['edges'] if edge['from_id'] not in door_ids and edge['to_id'] not in door_ids]

    args.record_dir = 'record_scratch/rec_good_test/multiBob_env_task_set_{}_{}'.format(args.num_per_apartment,
                                                                                        args.mode)
    executable_args = {
        'file_name': args.executable_file,
        'x_display': 0,
        'no_graphics': True
    }
    args.load_model = ('trained_models/env.virtualhome/task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/'
                      'agent.hrl_mcts_alice.False/mode.RL-algo.a2c-base.TF-gamma.0.95'
                      '-cclose.0.0-cgoal.0.0-lr0.0001-bs.32/stepmcts.24-lep.250-teleport.True-gtgraph/4200.pt')
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



    print('Number of episides: {}'.format(len(env_task_set)))

    agent_goal = 'full'
    args.task_type = 'full'
    # if args.task_type == 'put':
    #     agent_goal = 'put'
    num_agents = 1
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
                                    base_port=args.base_port,
                                    seed=None)
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
                           logging=True,
                           logging_graphs=True)

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
    a2c.load_model(args.load_model)

    test_results = []
    for i in range(28, 100):
        successes = []
        lengths = []
        for seed in range(5):
            try:
                for agent in arenas[0].agents:
                    agent.seed = seed
                res = a2c.eval(i)
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
                successes.append(finished)
                lengths.append(length)
                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(arenas[0].env.task_id,
                                                                                    info_results['task_name'],
                                                                                    seed)
                with open(log_file_name, 'wb') as flog:
                    pickle.dump(info_results, flog)
            except:
                arenas[0].reset_env()
        test_results.append({'S': successes, 'L': lengths})
    pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
    # pdb.set_trace()
