
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
comm.reset()
comm.add_character( 'Chars/Female1')
###################################
# Scene preparation
s, graph = comm.environment_graph()
id2node = {node['id']: node for node in graph['nodes']}
id_table = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
id_kitchencabinet = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencabinets'][0]
id_kitchendrawers = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounterdrawer']
id_cabinet = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cabinet'][0]
ids_in_table = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_table]
# Remove the ids from table
graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in ids_in_table and edge['to_id'] not in ids_in_table]
ids_remove = []
num_plates, num_glass = 0, 0
glass_ids = []
plate_ids = []
for idi in ids_in_table:
	if 'plate' in id2node[idi]['class_name'] and num_glass < 3:
		graph['edges'].append({'from_id': idi, 'to_id': id_cabinet, 'relation_type': 'INSIDE'})
		num_plates += 1
		plate_ids.append(idi)
	elif 'glass' in id2node[idi]['class_name'] and num_glass < 3:
		graph['edges'].append({'from_id': idi, 'to_id': id_kitchendrawers[0], 'relation_type': 'INSIDE'})
		num_glass += 1
		glass_ids.append(idi)
	else:
		ids_remove.append(idi)
graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in ids_remove and edge['to_id'] not in ids_remove]
graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in ids_remove]
success, gr = comm.expand_scene(graph)

wineglass = [299, 300]
recording = True
synth = 'normal'
###################################
try:
	comm.render_script(['<char0> [walk] <kitchencounterdrawer> ({})'.format(id_kitchendrawers[0])], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [open] <kitchencounterdrawer> ({})'.format(id_kitchendrawers[0])], recording=recording, image_synthesis=[synth], gen_vid=False)

	comm.render_script(['<char0> [walk] <kitchencabinets> ({})'.format(id_kitchencabinet)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [open] <kitchencabinets> ({})'.format(id_kitchencabinet)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [open] <kitchencounterdrawer> ({})'.format(id_kitchendrawers[0])], recording=recording, image_synthesis=[synth], gen_vid=False)


	comm.render_script(['<char0> [walk] <cabinet> ({})'.format(id_cabinet)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [open] <cabinet> ({})'.format(id_cabinet)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [grab] <plate> ({})'.format(plate_ids[0])], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [grab] <plate> ({})'.format(plate_ids[2])], recording=recording, image_synthesis=[synth], gen_vid=False)

	comm.render_script(['<char0> [walk] <kitchentable> ({})'.format(id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [put] <plate> ({}) <kitchentable> ({})'.format(plate_ids[0], id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [put] <plate> ({}) <kitchentable> ({})'.format(plate_ids[2], id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)

	ipdb.set_trace()
	comm.render_script(['<char0> [walk] <wineglass> ({})'.format(wineglass[0])], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [grab] <wineglass> ({})'.format(wineglass[0])], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [grab] <wineglass> ({})'.format(wineglass[1])], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [walk] <kitchentable> ({})'.format(id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)

	comm.render_script(['<char0> [put] <wineglass> ({}) <kitchentable> ({})'.format(wineglass[0], id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [put] <wineglass> ({}) <kitchentable> ({})'.format(wineglass[1], id_table)], recording=recording, image_synthesis=[synth], gen_vid=False)


	comm.render_script(['<char0> [walk] <tv> ({})'.format(427)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [switchon] <tv> ({})'.format(427)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [walk] <sofa> ({})'.format(369)], recording=recording, image_synthesis=[synth], gen_vid=False)
	comm.render_script(['<char0> [sit] <sofa> ({})'.format(369)], recording=recording, image_synthesis=[synth], gen_vid=False)

except:
	ipdb.set_trace()

#comm.render_script(['<char0> [watch] <tv> ({})'.format(427)], recording=recording, image_synthesis=[synth], gen_vid=False)

