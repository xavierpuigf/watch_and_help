import random
import pdb
import torch
import copy
import numpy as np
import time
import ray
import atexit

# @ray.remote
class ArenaMP(object):
    def __init__(self, arena_id, environment_fn, agent_fn):
        self.agents = []
        self.env_fn = environment_fn
        self.agent_fn = agent_fn
        self.arena_id = arena_id

        self.num_agents = len(agent_fn)

        print("Init Env")
        self.env = environment_fn(arena_id)
        for agent_type_fn in agent_fn:
            self.agents.append(agent_type_fn(arena_id, self.env))

        self.max_episode_length = self.env.max_episode_length

        atexit.register(self.close)

    def close(self):
        self.env.close()

    def get_port(self):
        return self.env.port_number


    def reset(self, task_id=None):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id)

        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                agent.reset(ob[it], self.env.python_graph, self.env.task_goal, seed=agent.seed)
            elif agent.agent_type == 'RL_MCTS':
                agent.reset(ob[it], self.env.python_graph, self.env.task_goal, seed=agent.seed)
            else:
                agent.reset(self.env.python_graph)

    def set_weigths(self, epsilon, weights):
        for agent in self.agents:
            if 'RL' in agent.agent_type:
                agent.epsilon = epsilon
                agent.actor_critic.load_state_dict(weights)

    def get_actions(self, obs, action_space=None):
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}
        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                opponent_subgoal = None
                if agent.recursive:
                    opponent_subgoal = self.agents[1 - it].last_subgoal
                # pdb.set_trace()
                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.get_goal(self.task_goal[it], self.env.agent_goals[it]), opponent_subgoal)
            elif 'RL' in agent.agent_type:
                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.goal_spec[it], action_space_ids=action_space[it])
        return dict_actions, dict_info

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

    def rollout_reset(self, logging=False, record=False):
        try:
            res = self.rollout(logging, record)
            return res
        except:
            print("Resetting...")
            self.env.close()
            self.env = self.env_fn(self.arena_id)


            for agent in self.agents:
                if 'RL' in agent.agent_type:
                    prev_eps = agent.epsilon
                    prev_weights = agent.actor_critic.state_dict()

            self.agents = []
            for agent_type_fn in self.agent_fn:
                self.agents.append(agent_type_fn(self.arena_id, self.env))

            self.set_weigths(prev_eps, prev_weights)
            return self.rollout(logging, record)

    def rollout(self, logging=0, record=False):
        t1 = time.time()
        self.reset()
        t2 = time.time()
        t_reset = t2 - t1
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []


        if logging > 0:
            info_rollout['pred_goal'] = []
            info_rollout['pred_close'] = []
            info_rollout['gt_goal'] = []
            info_rollout['gt_close'] = []
            info_rollout['mask_nodes'] = []

        if logging > 1:
            info_rollout['step_info'] = []
            info_rollout['script'] = []
            info_rollout['graph'] = []
            info_rollout['action_space_ids'] = []
            info_rollout['visible_ids'] = []
            info_rollout['action_tried'] = []

        rollout_agent = {}

        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            if 'RL' in agent.agent_type:
                rollout_agent[agent_id] = []

        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec[0]
            goal_class = list(pred.keys())[0].split('_')[1]
            id2node = {node['id']: node for node in init_graph['nodes']}
            info_goals = []
            info_goals.append([node for node in init_graph['nodes'] if node['class_name'] == goal_class])
            ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] == goal_class]
            info_goals.append([(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type'],
                                edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
            info_rollout['target'] = [pred, info_goals]


        agent_id = [id for id, enum_agent in enumerate(self.agents) if 'RL' in enum_agent.agent_type][0]
        while not done and nb_steps < self.max_episode_length:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step()
            if logging:
                curr_graph = env_info['graph']
                observed_nodes = agent_info[agent_id]['visible_ids']
                # pdb.set_trace()
                node_id = [node['bounding_box'] for node in obs[agent_id]['nodes'] if node['id'] == 1][0]
                edges_char = [(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type']) for edge in curr_graph['edges'] if edge['from_id'] == 1 and edge['to_id'] in observed_nodes]

                if logging > 0:
                    if 'pred_goal' in agent_info[agent_id].keys():
                        info_rollout['pred_goal'].append(agent_info[agent_id]['pred_goal'])
                        info_rollout['pred_close'].append(agent_info[agent_id]['pred_close'])
                        info_rollout['gt_goal'].append(agent_info[agent_id]['gt_goal'])
                        info_rollout['gt_close'].append(agent_info[agent_id]['gt_close'])
                        info_rollout['mask_nodes'].append(agent_info[agent_id]['mask_nodes'])

                if logging > 1:
                    info_rollout['step_info'].append((node_id, edges_char))
                    info_rollout['script'].append(agent_actions[agent_id])
                    info_rollout['action_tried'].append(agent_info[agent_id]['action_tried'])
                    info_rollout['graph'].append(curr_graph)
                    info_rollout['action_space_ids'].append(agent_info[agent_id]['action_space_ids'])
                    info_rollout['visible_ids'].append(agent_info[agent_id]['visible_ids'])

            nb_steps += 1
            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += reward
                # action_dict[agent_index] = agent_info[agent_index]['action']



            if record:
                actions.append(agent_actions)

            # append to memory
            for agent_id in range(self.num_agents):
                if 'RL' in self.agents[agent_id].agent_type and 'mcts_action' not in agent_info[agent_id]:
                    state = agent_info[agent_id]['state_inputs']
                    policy = [log_prob.data for log_prob in agent_info[agent_id]['probs']]
                    action = agent_info[agent_id]['actions']
                    rewards = reward

                    entropy_action.append(
                        -((agent_info[agent_id]['probs'][0] + 1e-9).log() * agent_info[agent_id]['probs'][0]).sum().item())
                    entropy_object.append(
                        -((agent_info[agent_id]['probs'][1] + 1e-9).log() * agent_info[agent_id]['probs'][1]).sum().item())
                    observation_space.append(agent_info[agent_id]['num_objects'])
                    action_space.append(agent_info[agent_id]['num_objects_action'])
                    last_agent_info = agent_info

                    rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action, rewards, 1))

        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[agent_id].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)
        info_rollout['t_reset'] = t_reset
        info_rollout['t_steps'] = t_steps

        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']


        info_rollout['env_id'] = self.env.env_id
        info_rollout['goals'] = list(self.env.task_goal[0].keys())
        # padding
        # TODO: is this correct? Padding that is valid?
        while nb_steps < self.max_episode_length:
            nb_steps += 1
            for agent_id in range(self.num_agents):
                if 'RL' in self.agents[agent_id].agent_type:
                    state = last_agent_info[agent_id]['state_inputs']
                    if 'edges' in obs.keys():
                        pdb.set_trace()
                    policy = [log_prob.data for log_prob in last_agent_info[agent_id]['probs']]
                    action = last_agent_info[agent_id]['actions']
                    rewards = reward
                    rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action, 0, 0))

        return c_r_all, info_rollout, rollout_agent


    def step(self):
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space)
        try:
            step_info = self.env.step(dict_actions)
        except:
            print("Time out for action: ", dict_actions)
            raise Exception
        return step_info, dict_actions, dict_info


    def run(self, random_goal=False, pred_goal=None):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """
        self.task_goal = copy.deepcopy(self.env.task_goal)
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        saved_info = {'task_id': self.env.task_id,
                      'env_id': self.env.env_id,
                      'task_name': self.env.task_name,
                      'gt_goals': self.env.task_goal[0],
                      'goals': self.task_goal,
                      'action': {0: [], 1: []},
                      'plan': {0: [], 1: []},
                      'subgoals': {0: [], 1: []},
                      # 'init_pos': {0: None, 1: None},
                      'finished': None,
                      'init_unity_graph': self.env.init_graph,
                      'goals_finished': [],
                      'belief': {0: [], 1: []},
                      'obs': []}
        success = False
        while True:
            (obs, reward, done, infos), actions, agent_info = self.step()
            success = infos['finished']
            if 'satisfied_goals' in infos:
                saved_info['goals_finished'].append(infos['satisfied_goals'])
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)
            for agent_id, info in agent_info.items():
                if 'belief_graph' in info:
                    saved_info['belief'][agent_id].append(info['belief_graph'])
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(info['plan'][:3])
                if 'subgoals' in info:
                    saved_info['subgoals'][agent_id].append(info['subgoals'][:3])
                if 'obs' in info:
                    saved_info['obs'].append(info['obs'])
            if done:
                break
        saved_info['finished'] = success
        return success, self.env.steps, saved_info