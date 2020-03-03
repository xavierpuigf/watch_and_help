import json
import pdb
import glob
import argparse
from collections import Counter

def detect_bad_placing(action_seq):
    actions = [x.split() if x is not None else [None, None, None] for x in action_seq]
    cont = 0
    for i in range(len(actions)-1):
        if (actions[i][0] is not None and
                'put' in actions[i][0] and actions[i+1][0] is not None and
                'grab' in actions[i+1][0] and actions[i][2] == actions[i+1][2]):
            cont += 1
    if cont > 0:
        return True
    return False

def detect_bad_walk(action_seq):
    action_seq = [x for x in action_seq if x is not None]
    if len(action_seq) == 0:
        return None

    ct = Counter(action_seq).most_common()
    if ct[0][1] > 40:
        return ct[0][0]
    return None


def add_to_stats(dict_apartments, content):
    env_id = content['env_id']
    if env_id not in dict_apartments:
        dict_apartments[env_id] = 0
    dict_apartments[env_id] += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analysis vids')
    parser.add_argument('--record_dir', default='../record/', type=str)
    args = parser.parse_args()

    dir_files = glob.glob('{}/init*50*'.format(args.record_dir))
    dict_failures = []
    destinations = []

    for dir_file in dir_files:
        if 'Bob' in dir_file:
            continue
                 
        dict_apartments = {}
        finished = 0
        count = 0
        errors = {
            'bad_placing': 0,
            'bad_walk': 0,
            'null': 0,
            'other': 0
        }
        print('\n', dir_file)
        print('======')
        json_files = glob.glob('{}/*.json'.format(dir_file))
        for json_file in json_files:
            if 'Bob' in json_file:
                continue
            count += 1
            try:
                with open(json_file, 'r') as f:
                    content = json.load(f)

            except:
                continue

            id2node = {node['id']: node for node in content['init_unity_graph']['nodes']}
            rooms = [node['id'] for node in content['init_unity_graph']['nodes'] if node['category'] == 'Rooms']
            if content['finished'] and len(content['action']['0']) > 50:
                if finished == 0:
                    print(len(content['action']['0']), json_file)

                finished += 1
                #print(dir_file)
                #pdb.set_trace()
                add_to_stats(dict_apartments, content)
            else:

                #try:
                no_errors = True
                actions = content['action']['0']
                if None in actions:
                    errors['null'] += 1
                    no_errors = False
                if detect_bad_placing(actions):
                    errors['bad_placing'] += 1
                    no_errors = False

                obj_walk = detect_bad_walk(actions)
                if obj_walk is not None:
                    obj_walk_curr = obj_walk.split('<')[1].split('>')[0]
                    obj_id = int(obj_walk.split('(')[1].split(')')[0])
                    if 'GRABBABLE' in id2node[obj_id]['properties']:
                        edge = [edge for edge in content['init_unity_graph']['edges']
                                if edge['from_id'] == obj_id and edge['to_id'] not in rooms]

                        if len(edge) == 0:
                            pass
                            #print(id2node[obj_id]['class_name'], content['env_id'])
                        else:
                            dest = id2node[edge[0]['to_id']]['class_name']
                            destinations.append(dest + str(content['env_id']))
                            #pdb.set_trace()

                    # if content['env_id'] == 6:
                    #     # if obj_walk_curr == 'kitchencabinets':
                    #     #     pdb.set_trace()
                    obj_and_env = 'WALK_' + obj_walk_curr + '_' + str(content['env_id'])

                    dict_failures.append(obj_and_env)

                    errors['bad_walk'] += 1
                    no_errors = False

                if no_errors:
                    add_to_stats(dict_apartments, content)
                    #pdb.set_trace()
                    errors['other'] += 1
                # except:
                #     pdb.set_trace()

        print(errors)

        print('({}/{}) ({:.2f})'.format(finished, count, finished*100./(count+1e-9)), count)
        for i in range(7):
            print(i, ': {}'.format(dict_apartments[i] if i in dict_apartments.keys() else 0))

    ct = Counter(dict_failures).most_common()
    print(ct)

    ct = Counter(destinations).most_common()
    print(ct)
