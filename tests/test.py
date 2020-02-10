
import sys
import matplotlib.pyplot as plt
import os
import ipdb
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')
from simulation.unity_simulator import comm_unity as comm_unity
# comm = comm_unity.UnityCommunication()
# comm.reset()
# comm.add_character( 'Chars/Female1')


import utils
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile
import pickle
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent
from envs.envs import UnityEnv

if __name__ == '__main__':
    num_agents = 2
    unity_env = UnityEnv(num_agents)

    # comm = comm_unity.UnityCommunication()
    #unity_env.unity_simulator.comm.render_script(['<char0> [walkforward]'], recording=False)
    unity_env.unity_simulator.comm.render_script(['<char0> [turnleft]'], recording=False, image_synthesis=[])

    ipdb.set_trace()
    obs = unity_env.get_observations()
    ipdb.set_trace()
