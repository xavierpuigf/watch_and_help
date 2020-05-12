import sys
sys.path.append('../virtualhome/')
sys.path.append('../vh_mdp/')
sys.path.append('../virtualhome/simulation/')
import pdb
import pickle
import os
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent
from arguments import get_args
from algos.arena import Arena
import pickle as pkl
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
import pickle as pkl
import json


def get_predicted_goal(episode_id):
    return


if __name__ == '__main__':
    args = get_args()
    # args.task = 'setup_table'
    # args.num_per_apartment = '50'
    # args.mode = 'full'
    # args.dataset_path = 'initial_environments/data/init_envs/init7_{}_{}_{}.pik'.format(args.task,
    #                                                                                        args.num_per_apartment,
    #                                                                                     args.mode)
    # data = pickle.load(open(args.dataset_path, 'rb'))

    pred_file = '/data/vision/torralba/ls-objectvideo/2icml2020/1virtualhome/vh_multiagent_models_goal_inference_video_eval/data/test_json_output_graph_sort_avg_insamelen_hid512_larger_largerv2_smallerv2_tranf_dp0_lstmavg_h2l1.p'
    with open(pred_file, 'rb') as f:
        predictions = pkl.load(f)
    with open('/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/match_demo_test.json', 'r') as f:
        match_demo_test = json.load(f)

    env_to_pred = {}
    for i in range(100):
        demo_env = match_demo_test[str(i)][0].replace('.pik', '')
        curr_pred = predictions[demo_env]['prediction']
        pred_dict = {}
        for p in curr_pred:
            if p != 'None':
                if p not in pred_dict:
                    pred_dict[p] = 0
                pred_dict[p] += 1
        env_to_pred[str(i)] = pred_dict

    args.max_episode_length = 250
    args.num_per_apartment = '20'
    args.base_port = 8088
    args.mode = 'check_neurips_test_recursive'
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/executables/exec_linux.04.27.x86_64'

    # env_task_set = pickle.load(open('initial_environments/data/init_envs/env_task_set_{}_{}.pik'.format(args.num_per_apartment, args.mode), 'rb'))
    env_task_set = pickle.load(open('initial_environments/data/init_envs/test_env_set_help_20_neurips.pik', 'rb'))

    for env in env_task_set:
        if env['env_id'] == 6:
            g = env['init_graph']
            door_ids = [302, 213]
            g['nodes'] = [node for node in g['nodes'] if node['id'] not in door_ids]
            g['edges'] = [edge for edge in g['edges'] if edge['from_id'] not in door_ids and edge['to_id'] not in door_ids]


    args.record_dir = 'record_scratch/rec_good_test/multiBob_env_task_set_{}_predgoal'.format(args.num_per_apartment)
    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    # env_task_set = []
    # for task_id, problem_setup in enumerate(data):
    #     env_id = problem_setup['apartment'] - 1
    #     task_name = problem_setup['task_name']
    #     init_graph = problem_setup['init_graph']
    #     goal = problem_setup['goal'][task_name]

    #     goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
    #                                           exclude=['cutleryknife'])
    #     print('env_id:', env_id)
    #     print('task_name:', task_name)
    #     print('goals:', goals)

    #     task_goal = {}
    #     for i in range(2):
    #         task_goal[i] = goals

    #     env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
    #                          'task_goal': task_goal,
    #                          'level': 0, 'init_rooms': [0, 0]})

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    #random.shuffle(episode_ids)
    episode_ids = sorted(episode_ids)
    num_tries = 5
    S = [[0]*5 for _ in range(len(episode_ids))]
    L = [[200]*5 for _ in range(len(episode_ids))]
    #seeds =
    test_results = {}

    def env_fn(env_id):
        return UnityEnvironment(num_agents=2,
                               max_episode_length=args.max_episode_length,
                               port_id=env_id,
                               env_task_set=env_task_set,
                               observation_types=[args.obs_type, args.obs_type],
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
    args_agent2 = {'agent_id': 2, 'char_index': 1}
    args_agent1.update(args_common)
    args_agent2.update(args_common)
    args_agent2.update({'recursive': False})
    agents = [lambda x, y: MCTS_agent(**args_agent1), lambda x, y: MCTS_agent(**args_agent2)]
    arena = ArenaMP(id_run, env_fn, agents)

    for iter_id in range(num_tries):
        # if iter_id > 0:
        #     test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(iter_id - 1), 'rb'))
        cnt = 0
        steps_list, failed_tasks = [], []
        for episode_id in episode_ids:

            curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(env_task_set[episode_id]['task_id'],
                                                                                     env_task_set[episode_id]['task_name'],
                                                                                     iter_id)
            if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
                test_results = {}
            else:
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))

            current_tried = iter_id
            if os.path.isfile(curr_log_file_name):
                with open(curr_log_file_name, 'rb') as fd:
                    file_data = pkl.load(fd)
                S[episode_id][current_tried] = file_data['finished']
                L[episode_id][current_tried] = max(len(file_data['action'][0]), len(file_data['action'][1]))
                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                continue
            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            is_finished = 0
            steps = 250
            try:

                arena.reset(episode_id)

                predicted_goal_class = env_to_pred[str(episode_id)]
                graph = arena.env.graph
                idnodes = {}
                for class_name in ['fridge', 'dishwasher', 'kitchentable', 'coffeetable', 'sofa']:
                    idnodes[class_name] = [node['id'] for node in graph['nodes'] if node['class_name'] == class_name][0]

                predicted_goal = {}
                for kpred, itv in predicted_goal_class.items():
                    spl = kpred.split('_')
                    if not spl[-1] in idnodes:
                        if spl[0] == 'sit':
                           target_name = '_'.join([spl[0], '1', spl[1]])
                        else:
                            target_name = '_'.join([spl[0], spl[1], '1'])
                    else:
                        id_target = idnodes[spl[-1]]
                        if spl[0] == 'sit':
                            spl[1] = '1'
                        target_name = '{}_{}_{}'.format(spl[0], spl[1], id_target)
                    predicted_goal[target_name] = itv
                # pdb.set_trace()
                original_goal = arena.env.task_goal[0]
                success, steps, saved_info = arena.run(pred_goal={0: original_goal, 1: predicted_goal})
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
                log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'],
                                                                                    saved_info['task_name'],
                                                                                    current_tried)

                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(log_file_name, 'wb'))
                else:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
            except:
                arena.reset_env()

            S[episode_id][current_tried] = is_finished
            L[episode_id][current_tried] = steps
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