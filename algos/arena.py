import random
import pdb
import copy

class Arena:
    def __init__(self, agent_types, environment):
        self.agents = []
        for agent_type in agent_types:
            self.agents.append(agent_type)
        self.num_agents = len(agent_types)
        self.env = environment

    def reset(self, task_id=None):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id)
        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                agent.reset(self.env.python_graph, self.env.task_goal, seed=it)
            else:
                agent.reset(self.env.python_graph)

    def get_actions(self, obs, action_space=None):
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}
        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                opponent_subgoal = None
                if agent.recursive:
                    opponent_subgoal = self.agents[1 - it].last_subgoal
                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.task_goal[it] if it == 0 else self.task_goal[it], opponent_subgoal)
            elif agent.agent_type == 'RL':
                # Goal encoding should be done with goal_spec

                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.goal_spec if it == 0 else self.task_goal[it], action_space_ids=action_space[it])
        return dict_actions, dict_info


    def step(self):
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space)
        # print(dict_actions)
        return self.env.step(dict_actions), dict_actions, dict_info


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
                      'goals': self.task_goal[0],
                      'action': {0: [], 1: []}, 
                      'plan': {0: [], 1: []},
                      'subgoal': {0: [], 1: []},
                      # 'init_pos': {0: None, 1: None},
                      'finished': None,
                      'init_unity_graph': self.env.init_unity_graph,
                      'obs': []}
        success = False
        while True:
            (obs, reward, done, infos), actions, agent_info = self.step()
            success = infos['finished']
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)
            for agent_id, info in agent_info.items():
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(info['plan'][:3])
                if 'subgoal' in info:
                    saved_info['subgoal'][agent_id].append(info['subgoal'][:3])
                if 'obs' in info:
                    saved_info['obs'].append(info['obs'])
            if done:
                break
        saved_info['finished'] = success
        return success, self.env.steps, saved_info
