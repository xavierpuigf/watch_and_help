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
#comm.render_script(['<char0> [close] <fridge> (288)'], image_synthesis=[], recording=False)
comm.render_script(['<char0> [walk] <wineglass> (193)'], image_synthesis=[], recording=False)
#comm.render_script(['<char0> [walk] <toilet> (37)', '<char0> [switchon] <toilet> (37)'], image_synthesis=[], recording=False)