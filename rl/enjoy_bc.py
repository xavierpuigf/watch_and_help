import scipy.signal
import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

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
    default='virtualhome_bc',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='trained_models/env.virtualhome_bc/task.find-numenvs.1-obstype.mcts-sim.unity/mode.BC-algo.a2c-attention.linear-gamma.0.99/',
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
parser.add_argument(
    '--num-steps',
    type=int,
    default=100,
    help='number of forward steps in A2C (default: 100)')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
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
            'behavior_cloning_train_file': ('/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent'
                                           '/tshu/vh_multiagent_models/record/init7_Bob_train_set')

    }

device = torch.device("cuda:0" if args.cuda else "cpu")
env = make_vec_envs(
    env_info,
    args.num_steps,
    args.simulator_type,
    args.seed + 1000,
    1,
    None,
    None,
    device,
    allow_early_resets=False,
    num_frame_stack=1)

# Get a render function
print(env)
print(args.load_dir)
# render_func = get_render_func(env)
render_func = None
# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
        torch.load(os.path.join(args.load_dir, "800.pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(1, 1).to(device)
# pdb.set_trace()
obs = env.reset()

import pdb
pdb.set_trace()
if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    with torch.no_grad():
        pdb.set_trace()
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det, action_indices=[plan_act, plan_obj])
    pdb.set_trace()
    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
