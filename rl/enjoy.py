import argparse
import os
import scipy.special
# workaround to unpickle olf model files
import sys

import numpy as np
import torch
import json
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--simulator-type',
    default='unity',
    choices=['unity', 'python'],
    help='whether to use unity or python sim')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='virtualhome',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    #default='trained_models/env.virtualhome_bc/task.find-numenvs.1-obstype.mcts-sim.unity/mode.BC-algo.a2c-attention.linear-gamma.0.99-lr1e-05/49800.pt',
    default='../../../tshu/vh_multiagent_models/rl/trained_models/env.virtualhome/task.complex-numenvs.1-obstype.mcts-sim.unity/algo.a2c-attention.dot-gamma.0.99/2300.pt',
    help='directory to save agent logs (default: ./trained_models/)')

parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')


parser.add_argument(
    '--executable_file', type=str, default='../../executables/exec_linux03.03/exec_linux03.3multiagent.x86_64')
parser.add_argument(
    '--base_port', type=int, default=8080)

parser.add_argument(
    '--display', type=str, default="2")

parser.add_argument(
    '--obs_type',
    type=str,
    default='mcts',
    choices=['full', 'rgb', 'visibleid', 'mcts'],
)
parser.add_argument(
    '--attention_type',
    type=str,
    default='linear',
    choices=['fc', 'dot', 'linear'],
    help='use a linear schedule on the learning rate')

parser.add_argument(
    '--model_type',
    default='TF',
    choices=['GNN', 'CNN', 'TF'],
    help='use a linear schedule on the learning rate')

args = parser.parse_args()

args.det = not args.non_det


env_info = {
            'env_name': args.env_name,
            'simulator_type': args.simulator_type,
            'task': 'complex',
            'testing': True,
            'base_port': args.base_port,
            'display': args.display,
            'executable_file': args.executable_file,
            'observation_type': args.obs_type,


    }

env = make_vec_envs(
    env_info,
    200,
    args.simulator_type,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False,
    num_frame_stack=1)

# Get a render function
print(env)
print(args.load_dir)
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
        torch.load(os.path.join(args.load_dir), map_location={'cuda:0': 'cpu'})

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
render_func = None
if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

import pdb

episodes_results = []

pdb.set_trace()
for count in range(140):
    steps_taken = 0
    current_actions = []
    for nsteps in range(200):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, info = env.step(action)
        current_actions.append(info[0]['actions'])
        steps_taken += 1
        if done.item():
            break

    episodes_results.append((done.item(), steps_taken, current_actions, env.envs[0].task_name))
    env.envs[0].count_test += 1
    obs = env.reset()

    with open('results_RL_scratch_log_2v.json', 'w+') as f:
        f.write(json.dumps(episodes_results, indent=4))

