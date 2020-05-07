import json
import pdb
from collections import Counter
import random
import pickle as pkl

all_p = []

home = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/'
with open(f'{home}/split/watch_scenes_split.json', 'r') as f:
    split = json.load(f)

with open(f'{home}/test_env_set_help_20_neurips.pik', 'rb') as f:
    data_help = pkl.load(f)

match_demo_test = {}
pred_str_to_demo = {}
for demo_test in split['test']:
    if demo_test['pred_str'] not in pred_str_to_demo:
        pred_str_to_demo[demo_test['pred_str']] = []
    pred_str_to_demo[demo_test['pred_str']].append(demo_test['file_name'].split('/')[-1])

for env_test in data_help:
    pred_str = env_test['pred_str']
    match_demo_test[env_test['task_id']] = pred_str_to_demo[pred_str]

with open(f'{home}/match_demo_test.json', 'w+') as f:
    f.write(json.dumps(match_demo_test, indent=4))