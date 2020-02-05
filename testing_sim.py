import gym
import ipdb
import sys
sys.path.append('../vh_mdp')
sys.path.append('../virtualhome')
import vh_graph
from simulation.unity_simulator import comm_unity as comm_unity
comm = comm_unity.UnityCommunication()
comm.reset()
comm.add_character()
comm.render_script(['<char0> [walk] <toilet> (37)'])