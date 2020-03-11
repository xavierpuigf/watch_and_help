import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')

from envs.unity_environment import UnityEnvironment
import pdb
import pickle
import random
from agents import MCTS_agent
from arguments import get_args
from arena import Arena
from utils import utils_goals

if __name__ == '__main__':
    args = get_args()
    args.task = 'setup_table'
    args.num_per_apartment = '50'
    args.mode = 'full'
    args.dataset_path = 'initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(args.task,
                                                                                        args.num_per_apartment,
                                                                                        args.mode)
    data = pickle.load(open(args.dataset_path, 'rb'))
    executable_args = {
            'file_name': args.executable_file,
            'x_display': 0,
            'no_graphics': True

    }

    env_task_set = []
    for task_id, problem_setup in enumerate(data):
        env_id = problem_setup['apartment'] - 1
        task_name = problem_setup['task_name']
        init_graph = problem_setup['init_graph']
        goal = problem_setup['goal'][task_name]

        goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
                                              exclude=['cutleryknife'])
        print('env_id:', env_id)
        print('task_name:', task_name)
        print('goals:', goals)

        task_goal = {}
        for i in range(2):
            task_goal[i] = goals

        env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
                             'task_goal': task_goal,
                             'level': 0, 'init_rooms': [0, 0]})

    episode_ids = list(range(len(env_task_set)))
    random.shuffle(episode_ids)

    env = UnityEnvironment(0, 0, 2, env_task_set, executable_args=executable_args)

    args_common = dict(unity_env=env,
                       max_episode_length=5,
                       num_simulation=100,
                       max_rollout_steps=3,
                       c_init=0.1,
                       c_base=1000000,
                       num_samples=1,
                       num_processes=1,
                       logging=True)

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent2 = {'agent_id': 2, 'char_index': 1}
    args_agent1.update(args_common)
    args_agent2.update(args_common)
    agents = [MCTS_agent(**args_agent1), MCTS_agent(**args_agent2)]
    arena = Arena(agents, env)
    arena.reset()


    arena.step()
    pdb.set_trace()

