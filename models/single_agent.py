from single_policy import SinglePolicy
class SingleAgent():
    def __init__(self, env, goal, agent_id):
        self.env = env
        self.goal = goal
        self.agent_id = agent_id
        self.policy_net = SinglePolicy()
    def reset(self):
        # TODO: check how do we reset the agent if the environment
        # is actually shared

    def obtain_actions(self, observations):
        # Returns a distribution of actions based on the policy
        return self.policy_net(observations)

    def obtain_beliefs(self):

    def update_beliefs(self):

    def obtain_observations(self):
        observations = None
        return observations
