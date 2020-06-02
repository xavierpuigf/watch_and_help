import glob
import pdb
import pickle
from tqdm import tqdm
file_path = '/data/vision/torralba//frames/data_acquisition/SyntheticStories/MultiAgent/challenge/vh_multiagent_models/record_scratch/rec_good_test/multiBob_env_task_set_20_check_neurips_test_recursive/*'

files = glob.glob(file_path)
for file_name in tqdm(files):
    with open(file_name, 'rb') as f:
        ct = pickle.load(f)
    if not 'action' in ct.keys():
        continue
    # if 'subgoal' not in ct:
    #     continue
    # if ct['task_name'] not in ['setup_table', 'put_dishwasher']: continue
    if ct['finished']: continue
    print(ct['env_id'], ct['task_id'])
    gt_goals = ct['gt_goals']
    bob_actions = ct['action'][1]
    if bob_actions[-1] is not None: continue

    alice_actions = ct['action'][0]
    
    # bob_subgoals = ct['subgoals'][1]
    # alice_subgoals = ct['subgoals'][0]
    
    # bob_obs = ct['obs'][1]
    # print(bob_actions)
    # for alice_action, bob_action, alice_sg, bob_sg in zip(alice_actions[-50:], bob_actions[-50:], alice_subgoals[-50:], bob_subgoals[-50:]):
    #     print(alice_action, alice_sg, bob_action, bob_sg)
    for alice_action, bob_action in zip(alice_actions[-50:], bob_actions[-50:]):
        print(alice_action, bob_action)
    pdb.set_trace()

    