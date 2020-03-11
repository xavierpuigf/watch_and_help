class Arena:
    def __init__(self, agent_types, environment):
        self.agents = []
        for agent_type in agent_types:
            self.agents.append(agent_type)

        self.env = environment

    def reset(self):
        self.env.reset()
        for it, agent in enumerate(self.agents):
            agent.reset(self.env.python_graph, self.env.task_goal, seed=it)

    def get_actions(self, obs):
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}
        for it, agent in enumerate(self.agents):
            dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.task_goal[it], op_subgoal[it])
        return dict_actions, dict_info


    def step(self):
        obs = self.env.get_observations()
        dict_actions, dict_info = self.get_actions(obs)
        self.env.step(dict_actions)

