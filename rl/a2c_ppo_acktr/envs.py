import os
import ipdb
import pdb
import gym
import numpy as np
import torch
import sys
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass




## ----------------------------------------------------------------------------
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])
print(home_path)
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')


import utils
from simulation.evolving_graph.utils import load_graph_dict
from profilehooks import profile
import pickle

from agents import MCTS_agent
from interface.envs.envs import UnityEnv
from interface.envs.envs_open_containers import UnityEnvOpenContainers
from interface.envs.envs_bc import UnityEnv as UnityEnvBC
## ----------------------------------------------------------------------------



def make_env(env_info, num_steps, simulator_type, seed, rank, log_dir, allow_early_resets):
    env_id = env_info['env_name']
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id == 'virtualhome':
            pdb.set_trace()
            data = pickle.load(open(home_path+'/vh_multiagent_models/initial_environments/data/init_envs/init1_10_same_room_simple.p', 'rb'))
            init_graph = data[0]['init_graph']

            id2node = {node['id']: node for node in init_graph['nodes']}
            print([(id2node[edge['from_id']]['class_name'], edge['from_id'], id2node[edge['to_id']]['class_name'], edge['to_id']) for edge in init_graph['edges'] if id2node[edge['from_id']]['class_name'] == 'wineglass'])
            # print([(id2node[edge['from_id']]['class_name'], edge['from_id'], id2node[edge['to_id']]['class_name'], edge['to_id']) for edge in init_graph['edges'] if edge['from_id'] == 115])
            # ipdb.set_trace()

            env_task_set = [{
                'env_id': 0,
                'task_id': 0,
                'task_name': 'setup_table',
                'init_graph': init_graph,
                'init_rooms': [76, 210],
                'level': 0,
                'task_goal': {agent_id: {'on_wineglass_235': 1} for agent_id in range(2)}
            }]

            # Only add graphics to the first instance
            simulator_args = {
                'file_name': env_info['executable_file'],
                'x_display': env_info['display'] if rank == 0 else None,
                'no_graphics': rank > 0

            }
            print(simulator_type)
            env = UnityEnv(num_agents=2, env_copy_id=rank, seed=rank, enable_alice=True, env_task_set=env_task_set,
                           task_type=env_info['task'], simulator_type=simulator_type, base_port=env_info['base_port'],
                           observation_type=env_info['observation_type'],
                           simulator_args=simulator_args,
                           max_episode_length=num_steps)


        elif env_id == 'virtualhome_bc':
            # Only add graphics to the first instance
            simulator_args = {
                'file_name': env_info['executable_file'],
                'x_display': env_info['display'] if rank == 0 else None,
                'no_graphics': rank > 0

            }
            env = UnityEnvBC(num_agents=2, env_copy_id=rank, seed=rank, enable_alice=True,
                           env_task_set_file=env_info['behavior_cloning_train_file'],
                           simulator_type=simulator_type, base_port=env_info['base_port'],
                           observation_type=env_info['observation_type'],
                           simulator_args=simulator_args,
                           max_episode_length=num_steps)


        elif env_id == 'virtualhome_opencontainers':
            data = pickle.load(open(home_path+'/vh_multiagent_models/initial_environments/data/init_envs/init7_setup_table_1_full.pik', 'rb'))
            init_graph = data[0]['init_graph']

            id2node = {node['id']: node for node in init_graph['nodes']}
            print([(id2node[edge['from_id']]['class_name'], edge['from_id'], id2node[edge['to_id']]['class_name'], edge['to_id']) for edge in init_graph['edges'] if id2node[edge['from_id']]['class_name'] == 'wineglass'])
            # print([(id2node[edge['from_id']]['class_name'], edge['from_id'], id2node[edge['to_id']]['class_name'], edge['to_id']) for edge in init_graph['edges'] if edge['from_id'] == 115])
            # ipdb.set_trace()

            env_task_set = [{
                'env_id': 0,
                'task_id': 0,
                'task_name': 'setup_table',
                'init_graph': init_graph,
                'init_rooms': [76, 210],
                'level': 0,
                'task_goal': {agent_id: {'on_cutleryfork_235': 1} for agent_id in range(2)}
            }]

            # Only add graphics to the first instance
            simulator_args = {
                'file_name': env_info['executable_file'],
                'x_display': env_info['display'] if rank == 0 else None,
                'no_graphics': rank > 0

            }
            print(simulator_type)
            env = UnityEnvOpenContainers(num_agents=2, env_copy_id=rank, seed=rank, enable_alice=True, env_task_set=env_task_set,
                           task_type=env_info['task'], simulator_type=simulator_type, base_port=env_info['base_port'],
                           observation_type=env_info['observation_type'],
                           simulator_args=simulator_args,
                           max_episode_length=num_steps)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)
        if type(env.observation_space) == gym.spaces.tuple.Tuple:
            obs_shape = env.observation_space[0].shape
        else:
            obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif not env_id.startswith('virtualhome') and len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        
        # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #     env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_info,
                  num_steps,
                  simulator_type,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    env_name = env_info if type(env_info) == str else env_info['env_name']
    if False: #env_name=='virtualhome':
        #    envs = [make_env(env_name, )UnityEnv(num_agents=1) for i in range(1)]
        if log_dir is not None:
            envs = bench.Monitor(
                    envs,
                    os.path.join(log_dir, str('1')),
                allow_early_resets=allow_early_resets)
    
    else:
        envs = [
                make_env(env_info, num_steps, simulator_type, seed, i, log_dir, allow_early_resets)
            for i in range(num_processes)
        ]

        if len(envs) > 1:
            envs = ShmemVecEnv(envs, context='fork')
        else:
            envs = DummyVecEnv(envs)

        if not env_name.startswith('virtualhome') and len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, ret=False)
            else:
                envs = VecNormalize(envs, gamma=gamma)

        envs = VecPyTorch(envs, device)

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        elif len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = {k: torch.from_numpy(obs[k]).float().to(self.device) for k in obs.keys()}
        return obs

    def step_async(self, actions):
        new_action_list = [] 
        for i, action in enumerate(actions):
            action = action.cpu()
            if isinstance(action, torch.LongTensor):
                # Squeeze the dimension for discrete actions
                action = action.squeeze(1)
            action = action.numpy()
            new_action_list.append(action)

        new_action_list = np.concatenate([ac[:, None] for ac in new_action_list], 1)
        self.venv.step_async(new_action_list)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {ob_id: torch.from_numpy(obs[ob_id]).float().to(self.device) for ob_id in obs.keys()}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        if type(wos) != gym.spaces.Dict:
            wos = [wos]
        self.shape_dim0 = {kwo: wo.shape[0] for kwo, wo in wos.spaces.items()}
        type_obs = {kwo: wo.dtype for kwo, wo in wos.spaces.items()}

        lows = {kwo: np.repeat(wo.low, self.nstack, axis=0) for kwo, wo in wos.spaces.items()}
        highs = {kwo: np.repeat(wo.high, self.nstack, axis=0) for kwo, wo in wos.spaces.items()}

        if device is None:
            device = torch.device('cpu')

        self.stacked_obs = {k: torch.zeros((venv.num_envs, ) + low.shape).to(device) for k, low in lows.items()}

        dict_observation = {}
        for key in wos.spaces.keys():
            dict_observation[key] = gym.spaces.Box(low=lows[key], high=highs[key], dtype=type_obs[key])

        observation_space = gym.spaces.Dict(dict_observation)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        for kob in self.stacked_obs.keys():
            tem = self.stacked_obs[kob][:, self.shape_dim0[kob]:].clone()
            self.stacked_obs[kob][:, :-self.shape_dim0[kob]] = tem

        for ob_type_id in self.stacked_obs.keys():
            for (i, new) in enumerate(news):
                if new:
                    self.stacked_obs[ob_type_id][i] = 0
            self.stacked_obs[ob_type_id][:, -self.shape_dim0[ob_type_id]:] = obs[ob_type_id]
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = {kob: torch.zeros(ob.shape) for kob, ob in self.stacked_obs.items()}
        else:
            for _, ob in self.stacked_obs.items():
                ob.zero_()

        for it in self.stacked_obs.keys():
            self.stacked_obs[it][:, -self.shape_dim0[it]:] = obs[it]
        return self.stacked_obs

    def close(self):
        self.venv.close()
