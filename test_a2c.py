import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.unity_environment import UnityEnvironment
import pdb
import pickle
import random
from agents import MCTS_agent, RL_agent
from arguments import get_args
from algos.arena import Arena
from algos.a2c import A2C
from utils import utils_goals, utils_rl_agent

if __name__ == '__main__':
    args = get_args()
    args.task = 'setup_table'
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
        env_task_set['init_rooms'] = ['kitchen']
        env_task_set['task_goal'] = {0: {single_goal: 1}, 1: {single_goal: 1}}
        env_task_set = [env_task_set]

    # env_task_set = []
    # for task_id, problem_setup in enumerate(data):
    #     env_id = problem_setup['apartment'] - 1
    #     task_name = problem_setup['task_name']
    #     init_graph = problem_setup['init_graph']
    #     goal = problem_setup['goal'][task_name]
    #
    #     goals = utils_goals.conv ert_goal_spec(task_name, goal, init_graph,
    #                                           exclude=['cutleryknife'])
    #     print('env_id:', env_id)
    #     print('task_name:', task_name)
    #     print('goals:', goals)
    #
    #     task_goal = {}
    #     for i in range(2):
    #         task_goal[i] = goals
    #
    #     env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
    #                          'task_goal': task_goal,
    #                          'level': 0, 'init_rooms': [0, 0]})

    episode_ids = list(range(len(env_task_set)))
    random.shuffle(episode_ids)



    env = UnityEnvironment(num_agents=num_agents, max_episode_length=args.max_episode_length,
                           env_task_set=env_task_set,
                           agent_goals=['grab'],
                           observation_types=[args.obs_type],
                           use_editor=args.use_editor,
                           executable_args=executable_args)

    args_mcts = dict(unity_env=env,
                       recursive=True,
                       max_episode_length=5,
                       num_simulation=100,
                       max_rollout_steps=3,
                       c_init=0.1,
                       c_base=1000000,
                       num_samples=1,
                       num_processes=1,
                       logging=True)


    graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args.max_num_objects,
                                              max_num_edges=args.max_num_edges, current_task=None, simulator_type='unity')

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent2 = {'agent_id': 1, 'char_index': 0,
                   'args': args, 'graph_helper': graph_helper}
    args_agent1.update(args_mcts)
    #args_agent2.update(args_common)
    #agents = [MCTS_agent(**args_agent1), RL_agent(**args_agent2)]
    agents = [RL_agent(**args_agent2)]
    arena = A2C(agents, env, args)
    arena.train()
    pdb.set_trace()

    arena.run()

