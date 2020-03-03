import pickle
import glob
import json
import pdb
import collections

if __name__ == '__main__':
    file_name_path = '../initial_environments/data/init_envs/*' #init7_read_book_50_simple.pik'
    file_names = glob.glob(file_name_path)
    for file_name in file_names:
        print('\n')
        print(file_name)
        print('=========')
        try:
            data = pickle.load(open(file_name, 'rb'))
            object_dict = {}
            for elem in data:
                tasks = list(elem['goal'].values())[0]
                objects = [list(task.keys())[0].split('_')[1] for task in tasks]
                graph = elem['init_graph']
                id2node = {node['id']: node for node in graph['nodes']}

                for edge in graph['edges']:
                    if edge['relation_type'] != 'INSIDE':
                        continue
                    class_name = id2node[edge['from_id']]['class_name']
                    if class_name in objects:
                        if class_name not in object_dict:
                            object_dict[class_name] = []
                        object_dict[class_name].append(id2node[edge['to_id' ]]['class_name'])
                #pdb.set_trace()
            for obj in object_dict.keys():
                cnt = collections.Counter(object_dict[obj])
                print(obj, cnt)
        except:
            continue
