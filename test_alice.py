import sys
import os
from ray.util import ActorPool

sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')
import pdb
import pickle
import json
import random
# import ray
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals


import random

class MCTSArena(ArenaMP):

    def run_and_save(self, episode_id, record_dir):
        self.reset(episode_id)
        success, steps, saved_info = self.run()

        Path(record_dir).mkdir(parents=True, exist_ok=True)
        if len(saved_info['obs']) > 0:
            pickle.dump(saved_info, open(record_dir + '/logs_agent_{}_{}.pik'.format(saved_info['task_id'], saved_info['task_name']), 'wb'))
        else:
            with open(record_dir + '/logs_agent_{}_{}.json'.format(saved_info['task_id'], saved_info['task_name']), 'w+') as f:
                f.write(json.dumps(saved_info, indent=4))

        is_finished = 1 if success else 0
        return (episode_id, is_finished, steps)

if __name__ == '__main__':
    args = get_args()
    # ray.init()
    # MCTSArena = ray.remote(MCTSArena)
    # args.task = 'setup_table'
    args.max_episode_length = 250
    args.num_per_apartment = 10
    args.mode = 'check_neurips_test_multiple2'
    # args.dataset_path = 'initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(args.task,
    #                                                                                        args.num_per_apartment,
    #                                                                                     args.mode)
    # data = pickle.load(open(args.dataset_path, 'rb'))
    # args.record_dir = 'record/Alice_test_set_30'
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.27.x86_64'

    # env_task_set = pickle.load(open('initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'rb'))
    env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_help_10_multitask_neurips.pik', 'rb'))
    #env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_help_20_neurips.pik', 'rb'))

    if args.use_editor:
        env_task_set = [env_task_set[q] for q in [82]]
        # args.recording = True

    # Filter out the rug and door
    for env in env_task_set:
        if env['env_id'] == 6:
            g = env['init_graph']
            door_ids = [302, 213]
            g['nodes'] = [node for node in g['nodes'] if node['id'] not in door_ids]
            g['edges'] = [edge for edge in g['edges'] if edge['from_id'] not in door_ids and edge['to_id'] not in door_ids]


    args.record_dir = 'record_scratch/rec_good_test/multiAlice_env_task_set_{}_{}'.format(args.num_per_apartment, args.mode)
    # args.record_dir = 'record_scratch/rec_good/Alice_env_task_set_{}_{}'.format(args.num_per_apartment, args.mode)

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
    #seeds =
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
    # args_agent2 = {'agent_id': 2, 'char_index': 1}
    args_agent1.update(args_common)
    # args_agent2.update(args_common)
    # args_agent2.update({'recursive': True})
    agents = [lambda x, y: MCTS_agent(**args_agent1)]

    # num_proc = 1
    # arenas = [MCTSArena.remote(it, env_fn, agents) for it in range(num_proc)]
    # [ray.get(arena.reset.remote(0)) for arena in arenas]
    # pool = ActorPool(arenas)
    # res = pool.map(lambda actor, index: actor.run_and_save.remote(index, args.record_dir), [0,1])
    # # pool.join()
    # pdb.set_trace()
    #
    # ray.shutdown()


    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    # episode_ids = [656]
    # g = env_task_set[3351]['init_graph']
    # can_id = [node['id'] for node in env_task_set[3351]['init_graph']['nodes'] if 'garbage' in node['class_name']][0]
    # env_task_set[3351]['init_graph']['nodes'] = [node for node in g['nodes'] if node['id'] != can_id]
    # env_task_set[3351]['init_graph']['edges'] = [edge for edge in g['edges'] if edge['from_id'] != can_id and edge['to_id'] != can_id]
    for iter_id in range(4, 5):
        #if iter_id > 0:

        cnt = 0
        steps_list, failed_tasks = [], []

        for episode_id in episode_ids:
            if episode_id < 41:
                continue
            if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
                test_results = {}
            else:
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))

            if episode_id in test_results:
                if len(test_results[episode_id]['S']) > num_tries:
                    continue
                else:
                    current_tried = len(test_results[episode_id]['S'])
            else:
                current_tried = 0
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
        # print('AL:', np.array(L).mean())
        # print('SR:', np.array(S).mean())
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

