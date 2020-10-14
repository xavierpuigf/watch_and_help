import gym
import scipy.special
import ipdb
import sys
import json
import random
import numpy as np
import pdb
import timeit
import json
import os
import argparse
import glob

import pickle


def get_metrics_reward(alice_results, test_results, episode_ids):
    mS = []
    mL = []
    mSwS = []
    # pdb.set_trace()
    for seed in range(5):

        alice_S = []
        alice_L = []
    normalized_by_suc = False
    for episode_id in episode_ids:
        Ls = []
        Ss = []
        SWSs = []
        L_A_seeds = []
        for seed_alice in range(5):
            if episode_id not in alice_results:
                S_A, L_A = 0, 250
                pdb.set_trace()
                # continue
            else:
                if alice_results[episode_id]['S'][seed_alice] == '':
                    print(episode_id, seed)
                    continue

                S_A = alice_results[episode_id]['S'][seed_alice]
                L_A = alice_results[episode_id]['L'][seed_alice]
                L_A_seeds.append(L_A)
        if episode_id not in test_results:
            print(episode_id, seed)
            continue
        L_A_seeds = [t for t in L_A_seeds if t is not None]
        if normalized_by_suc:
            L_A_seeds = np.mean([t for t in L_A_seeds if t < 250])
        else:
            L_A_seeds = np.mean(L_A_seeds)

        Ls = []
        Ss = []
        for seed_bob in range(5):
            try:
                if test_results[episode_id]['S'][seed_bob] == '':
                    print(episode_id, seed)
                    continue
                if test_results[episode_id]['S'][seed_bob] is None:
                    print(episode_id, seed)
                    continue
                S_B = test_results[episode_id]['S'][seed_bob]
                L_B = test_results[episode_id]['L'][seed_bob]
                if L_B == 250:
                    S_B = 0.
            except:
                pdb.set_trace()

            Ls.append(L_B)
            Ss.append(S_B)

        Ls = [t for t in Ls if t is not None]
        if normalized_by_suc:
            Ls = [t for t in Ls if t < 250]
        if len(Ls) > 0:
            # if len([t for t in Ss if t == 0.]) > 0:
            #     pdb.set_trace()
            SWSs.append(np.mean([-ls * 1./250 + sb for ls, sb in zip(Ls, Ss)]))
            # mSwS.append(SWSs)
            # if SWSs > 0:
            #     cont_better += 1
            #

            # pdb.set_trace()
            # print(episode_id)
        mS.append(np.mean(Ss))
        mL.append(np.mean(Ls))
        mSwS.append(np.mean(SWSs))
    # print('Alice:', np.mean(alice_S), np.mean(alice_L))
    # print('Alice:', np.mean(alice_S), '({})'.format(np.std(alice_S)), np.mean(alice_L), '({})'.format(np.std(alice_L)))
    # print('Bob:', np.mean(Ss), '({})'.format(np.std(Ss)), np.mean(Ls), '({})'.format(np.std(Ls)), np.mean(SWSs), '({})'.format(np.std(SWSs)))

    ns = np.sqrt(len(mS))
    nw = np.sqrt(len(mSwS))
    return np.mean(mS), np.mean(mL), np.mean(mSwS), np.std(mS)/ns, np.std(mL)/ns,  np.std(mSwS)/nw


parser = argparse.ArgumentParser()


if __name__ == '__main__':
    args = parser.parse_args()
    print (' ' * 26 + 'Options')
    for k, v in vars(args).items():
            print(' ' * 26 + k + ': ' + str(v))
    env_task_set = pickle.load(open('../dataset/test_env_set_help.pik', 'rb'))
    
    args.record_dir_alice = '../test_results/multiAlice_env_task_set_20_hp'
    alice_results = pickle.load(open(args.record_dir_alice + '/results.pkl'.format(0), 'rb'))


    record_dirs = [
            '../test_results/multiBob_env_task_set_20_random_action',
        '../test_results/multiBob_env_task_set_20_hp_randomgoal',
        '../test_results/multiBob_env_task_set_20_hp_predgoal',  ###
        '../test_results/multiBob_env_task_set_20_hp_truegoal',
        '../test_results/multiBob_env_task_set_20_hybrid',
        '../test_results/multiBob_env_task_set_20_hybrid_predgoal',
        '../test_results/multiBob_env_task_set_20_hrl_truegoal',  #
        '../test_results/multiBob_env_task_set_20_hrl_predgoal',  #
        '../test_results/multiAlice_env_task_set_20_hp',

    ]
    task_names = ['setup_table', 'put_fridge', 'prepare_food', 'put_dishwasher', 'read_book']
    final_results = {'S': {}, 'SWS': {}, 'L': {}, 'classes': task_names}
    for record_dir in record_dirs:
        test_results = pickle.load(open(record_dir + '/results.pkl', 'rb'))
        method_name = record_dir.split('20_')[-1]
        final_results['S'][method_name] = [], []
        final_results['SWS'][method_name] = [], []
        final_results['L'][method_name] = [], []
        num_agents = 1

        episode_ids = list(range(len(env_task_set)))
        S = [0] * len(episode_ids)
        L = [200] * len(episode_ids)
        SRO, ALO, SWSO, stdRO, stdLO, stdSO = get_metrics_3(alice_results, test_results, episode_ids)
        #print('overall:', SRO, ALO, SWSO)



        sr_list, al_list, sws_list = [], [], []
        for task_name in task_names:
            episode_ids_task = [episode_id for episode_id in episode_ids if env_task_set[episode_id]['task_name'] == task_name]
            SR, AL, SWS, stdR, stdL, stdS = get_metrics_3(alice_results, test_results, episode_ids_task)
            sr_list.append(str(SR))
            al_list.append(str(AL))
            sws_list.append(str(SWS))

            final_results['S'][method_name][0].append(SR)
            final_results['SWS'][method_name][0].append(SWS)
            final_results['L'][method_name][0].append(AL)
            final_results['S'][method_name][1].append(stdR)
            final_results['SWS'][method_name][1].append(stdS)
            final_results['L'][method_name][1].append(stdL)



            # print('{}:'.format(task_name), SR, AL, SWS)
        final_results['S'][method_name][0].append(SRO)
        final_results['SWS'][method_name][0].append(SWSO)
        final_results['L'][method_name][0].append(ALO)
        final_results['S'][method_name][1].append(stdRO)
        final_results['SWS'][method_name][1].append(stdSO)
        final_results['L'][method_name][1].append(stdLO)

        sr_list.append(str(SRO))
        al_list.append(str(ALO))
        sws_list.append(str(SWSO))

        print("SR")
        print(','.join(sr_list))
        print("AL")
        print(','.join(al_list))
        print("Reward")
        print(','.join(sws_list))
        with open('results_mcts_across_seeds_all_baselines_reward.json', 'w+') as f:
            f.write(json.dumps(final_results))
