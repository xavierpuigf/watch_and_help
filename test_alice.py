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
import ray
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals


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
    # args.num_per_apartment = '50'
    # args.mode = 'full'
    # args.dataset_path = 'initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(args.task,
    #                                                                                        args.num_per_apartment,
    #                                                                                     args.mode)
    # data = pickle.load(open(args.dataset_path, 'rb'))
    # env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_30.pik', 'rb'))
    # args.record_dir = 'record/Alice_test_set_30'
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.18.x86_64'
    env_task_set = pickle.load(open('initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'rb'))
    args.record_dir = 'record_scratch/Alice_env_task_set_{}_{}'.format(args.num_per_apartment, args.mode)
    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    id_run = 5
    episode_ids = list(range(len(env_task_set)))
    random.seed(id_run)
    random.shuffle(episode_ids)
    S = [0] * len(episode_ids)
    L = [200] * len(episode_ids)
    test_results = {}

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
                         logging_graphs=False)

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


    arena = ArenaMP(id_run, env_fn, agents)

    for iter_id in range(1):
        #if iter_id > 0:

        cnt = 0
        steps_list, failed_tasks = [], []
        for episode_id in episode_ids:
            if episode_id not in [1424]:
                continue
            if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(iter_id)):
                test_results = {}
            else:
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(iter_id), 'rb'))
            if episode_id in test_results and test_results[episode_id]['S'] > 0: continue
            print('episode:', episode_id)
            # try:
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
                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(args.record_dir + '/logs_agent_{}_{}.pik'.format(saved_info['task_id'], saved_info['task_name']), 'wb'))
                else:
                    with open(args.record_dir + '/logs_agent_{}_{}.json'.format(saved_info['task_id'], saved_info['task_name']), 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
            else:
                pass

            S[episode_id] = is_finished
            L[episode_id] = steps
            test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(iter_id), 'rb'))
            test_results[episode_id] = {'S': is_finished,
                                        'L': steps}
            pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(iter_id), 'wb'))

        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        print('AL:', np.array(L).mean())
        print('SR:', np.array(S).mean())
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(iter_id), 'wb'))

