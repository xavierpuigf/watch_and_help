import pickle as pkl
import json

import os 
curr_dir = os.path.dirname(os.path.realpath(__file__))

pred_file = f'{curr_dir}/../dataset/watch_pred.p'
with open(f'{curr_dir}/../dataset/match_demo_test.json', 'r') as f:
        match_demo_test = json.load(f)

with open(pred_file, 'rb') as f:
    predictions = pkl.load(f)

env_to_pred = {}
for i in range(len(match_demo_test)):
    demo_env = match_demo_test[str(i)][0].replace('.pik', '')
    curr_pred = predictions[demo_env]['prediction']
    pred_dict = {}
    for p in curr_pred:
        if p != 'None':
            if p not in pred_dict:
                pred_dict[p] = 0
            pred_dict[p] += 1
    env_to_pred[str(i)] = pred_dict
