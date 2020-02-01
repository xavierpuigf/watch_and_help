import gym
import ipdb
import sys
sys.path.append('../vh_mdp')
sys.path.append('../virtualhome')

from simulation.unity_simulator import comm_unity as comm_unity

comm = comm_unity.UnityCommunication()
comm.reset(0)
comm.add_character()



#comm.render_script(['<char0> [open] <fridge> (288)'], image_synthesis=[], recording=False)
for _ in range(5):
	s, message = comm.render_script(['<char0>  [walktowards] <bathroom> (1)'] , image_synthesis=[], recording=False)
	#s, message = comm.render_script(['<char0>  [walktowards] <wineglass> (193)'] , image_synthesis=[], recording=False)
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