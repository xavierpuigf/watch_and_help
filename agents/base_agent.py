from pathlib import Path
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing

import vh_graph
from vh_graph.envs import belief
from vh_graph.envs.vh_env import VhGraphEnv


class BaseAgent:
    """
    Base agent class
    """
    def __init__(self, env, sim_env, max_episode_length):
        self.env = env
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.previous_belief_graph = None


    def sample_belief(self, obs_graph):
        self.belief.update_from_gt_graph(obs_graph)
        if self.previous_belief_graph is None:
            self.belief.reset_belief()
            new_graph = self.belief.sample_from_belief()
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph
        else:
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph


    def get_action(self, graph, task_goal):
        return self.env.get_action_space()[0], {}



    def rollout(self, graph, task_goal):
        nb_steps = 0
        done = False
        _ = self.env.reset(graph, task_goal)     
        self.env.to_pomdp()
        gt_state = self.env.vh_state.to_dict()
        self.belief = belief.Belief(gt_state)
        self.sample_belief(self.env.get_observations(0))
        self.sim_env.reset(self.previous_belief_graph, task_goal)
        self.sim_env.to_pomdp()

        while not done and nb_steps < self.max_episode_length:
            action = self.get_action()

            # Take a step in the belief and in the environment
            reward, state, infos = self.env.step({0: action})
            _, _, _ = self.sim_env.step({0: action})
            self.env.get_observations(0)
            self.sample_belief(self.env.get_observations(0))
            self.sim_env.reset(self.previous_belief_graph, task_goal)
            nb_steps += 1


            
 