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
comm.reset(0)
comm.add_character()

s, graph = comm.environment_graph()
ipdb.set_trace()
# [node for node in graph['nodes'] if node['class_name'] == 'character']
comm.render_script(['<char0> [walk] <fridge> (298)'], image_synthesis=[], recording=False, processing_time_limit=200)
comm.render_script(['<char0> [open] <fridge> (298)'], image_synthesis=[], recording=False)

ipdb.set_trace()
# comm.render_script(['<char0>  [walk] <wineglass> (193)'] , image_synthesis=[], recording=False)

for _ in range(9):
	#s, message = comm.render_script(['<char1>  [walktowards] <bathroom> (1)'] , image_synthesis=[], recording=False)
	s, message = comm.render_script(['<char0>  [walktowards] <cabinet> (408)'] , image_synthesis=[], recording=False)
	print(s, message)
#comm.render_script(['<char0>  [walktowards] <wineglass> (193)'] , image_synthesis=[], recording=False)
#comm.render_script(['<char0>  [grab] <wineglass> (193)'], image_synthesis=[], recording=False)
#_, message = comm.render_script(['<char0>  [grab] <wineglass> (192)'], image_synthesis=[], recording=False)
print(message)
#comm.render_script(['<char0> [walk] <wineglass> (193)'], image_synthesis=[], recording=False)
#comm.render_script(['<char0> [walk] <toilet> (37)', '<char0> [switchon] <toilet> (37)'], image_synthesis=[], recording=False)

# step 0009:	|"system": [walk] <wineglass> (193)       						 |"my_agent": [walk] <wineglass> (193)
# step 0010:	|"system": [grab] <wineglass> (193)       						 |"my_agent": [grab] <wineglass> (193)
# step 0011:	|"system": [grab] <wineglass> (192)