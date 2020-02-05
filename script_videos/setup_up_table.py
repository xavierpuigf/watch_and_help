# set up table
import sys
import os
import ipdb
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')
from simulation.unity_simulator import comm_unity as comm_unity

comm = comm_unity.UnityCommunication()
comm.add_character()
s, graph = comm.environment_graph()
ipdb.set_trace()
id2node = {node['id']: node for node in graph['nodes']}

id_table = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
id_kitchencabinet = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencabinets'][0]
id_cabinet = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cabinet'][0]

ids_in_table = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_table]


# Remove the ids from table
graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in ids_in_table and edge['to_id'] not in ids_in_table]


ids_remove = []
num_plates, num_glass = 0, 0
for idi in ids_in_table:
	if 'plate' in id2node[idi]['class_name'] and num_glass < 3:
		graph['edges'].append({'from_id': idi, 'to_id': id_cabinet, 'relation_type': 'INSIDE'})
		num_plates += 1
	elif 'glass' in id2node[idi]['class_name'] and num_glass < 3:
		graph['edges'].append({'from_id': idi, 'to_id': id_kitchencabinet, 'relation_type': 'INSIDE'})
		num_glass += 1
	else:
		ids_remove.append(idi)

graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in ids_remove and edge['to_id'] not in ids_remove]
graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in ids_remove]
success, gr = comm.expand_scene(graph)
ipdb.set_trace()