import json
import pdb
from collections import Counter
import random
all_p = []
with open('info_demo_scenes_posteccv.json', 'r') as f:
    content = json.load(f)
for task_name, predicates_list in content['task_name_to_predicates'].items():
    predicates = list(set(predicates_list))
    print(task_name)
    envs = []
    curr_all_p = []
    for pred in predicates:
        progs = content['goal_dict'][pred]

        envs += [x[1] + 1 for x in progs[1]]
        all_p += [x[0] for x in progs[1]]
        curr_all_p += [x[0] for x in progs[1]]
    cnt = Counter(envs)
    print(len(curr_all_p))
    print(cnt)
len(all_p)

file_name_json = 'watch_scenes_split.json'

train_files, test_files = [], []
train_pred, test_pred = [], []
for task_name, predicates_list in content['task_name_to_predicates'].items():
    predicates = list(set(predicates_list))
    random.shuffle(predicates)
    test_predicates = predicates[:20]
    train_predicates = predicates[20:]
    train_pred += train_predicates
    test_pred += test_predicates

    for pred in train_predicates:
        progs_info = content['goal_dict'][pred]
        pred_dict = progs_info[0]
        programs = progs_info[1]
        for program in programs:
            env_id = program[1]
            file_name = program[0]

            train_files.append({'file_name': file_name, 'pred_dict': pred_dict, 'pred_str': pred, 'env_id': env_id, 'task_name': task_name})

    for pred in test_predicates:
        progs_info = content['goal_dict'][pred]
        pred_dict = progs_info[0]
        programs = progs_info[1]
        for program in programs:
            env_id = program[1]
            file_name = program[0]

            test_files.append({'file_name': file_name, 'pred_dict': pred_dict, 'pred_str': pred, 'env_id': env_id,
                                'task_name': task_name})

# pdb.set_trace()
watch_scene_split = {'train': train_files,
                     'test': test_files,
                     'train_pred': train_pred,
                     'test_pred': test_pred}

with open(file_name_json, 'w+') as f:
    f.write(json.dumps(watch_scene_split, indent=4))

print(len(train_files), len(test_files))
print(len(set(train_pred)), len(set(test_pred)))