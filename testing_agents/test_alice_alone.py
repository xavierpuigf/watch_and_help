import sys
import os
from ray.util import ActorPool
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals





if __name__ == '__main__':
    args = get_args()

    args.max_episode_length = 250
    args.num_per_apartment = 10
    args.dataset_path = './dataset/test_env_set_help_20_neurips.pik'
    
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))


    args.record_dir = '../test_results/multiAlice_env_task_set_{}_{}'.format(args.num_per_apartment, args.mode)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    #random.shuffle(episode_ids)
    episode_ids = sorted(episode_ids)
    num_tries = 5
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]
    
    test_results = {}

    if args.use_editor:
        num_tries = 1

    def env_fn(env_id):
        return UnityEnvironment(num_agents=1,
                                max_episode_length=args.max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                observation_types=[args.obs_type],
                                use_editor=args.use_editor,
                                executable_args=executable_args,
                                base_port=args.base_port)


    args_common = dict(recursive=False,
                         max_episode_length=5,
                         num_simulation=100,
                         max_rollout_steps=5,
                         c_init=0.1,
                         c_base=1000000,
                         num_samples=1,
                         num_processes=1,
                         logging=True,
                         logging_graphs=True)

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent1.update(args_common)
    agents = [lambda x, y: MCTS_agent(**args_agent1)]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    for iter_id in range(5):
        #if iter_id > 0:

        cnt = 0
        steps_list, failed_tasks = [], []
        current_tried = iter_id
        episode_ids = [episode_ids[0]]
        for episode_id in episode_ids:
            if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
                test_results = {}
            else:
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))

            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2
            # continue
            # try:
            print(env_task_set[episode_id]['task_name'])
            
            if True:
                
                arena.reset(episode_id)
                success, steps, saved_info = arena.run()
                print('-------------------------------------')
                print('success' if success else 'failure')
                print('steps:', steps)
                print('-------------------------------------')
                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                is_finished = 1 if success else 0

                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'], saved_info['task_name'], current_tried)
                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(log_file_name, 'wb'))
                else:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
            else:
                arena.reset_env()

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)

            if os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))
            test_results[episode_id] = {'S': S[episode_id],
                                        'L': L[episode_id]}
            pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

