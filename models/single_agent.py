from .single_policy import SinglePolicy
from torch import distributions
import torch
import pdb

class SingleAgent():
    def __init__(self, env, goal, agent_id):
        self.env = env
        self.goal = goal
        self.agent_id = agent_id
        self.policy_net = SinglePolicy()
        self.beliefs = None

    def reset(self):
        # TODO: check how do we reset the agent if the environment
        # is actually shared
        pass

    def get_action(self, actions):
        # Returns a distribution of actions based on the policy
        logits = torch.tensor([1.]*len(actions))
        distr = distributions.categorical.Categorical(logits=logits)
        return distr

    def get_first_obj(self, observations):
        logits = torch.tensor([1.]*len(observations['nodes']))
        distr = distributions.categorical.Categorical(logits=logits)
        return distr

    def get_second_obj(self, triples):
        logits = torch.tensor([1.]*len(triples))
        distr = distributions.categorical.Categorical(logits=logits)
        return distr

    def get_beliefs(self):
        return self.beliefs

    def update_beliefs(self, beliefs):
        self.beliefs = beliefs

    def get_observations(self):
        return self.env.get_observations()

    def get_instruction(self, observations):
        distr_o1 = self.get_first_obj(observations)
        obj1_id = distr_o1.sample()
        object_1_selected = observations['nodes'][obj1_id]

        # Decide action
        action_candidates_tripl = self.env.get_action_space(obj1=object_1_selected, structured_actions=True)
        actions_unique = list(set([x[0] for x in action_candidates_tripl]))
        distr_a1 = self.get_action(actions_unique)
        action_selected = actions_unique[distr_a1.sample()]
        action_candidates_tripl = self.env.get_action_space(action=action_selected,
                                                            obj1=object_1_selected,
                                                            structured_actions=True)
        if len(action_candidates_tripl) == 1:
            id_triple = 0

        else:
            distr_triple = self.get_second_obj(action_candidates_tripl)
            id_triple = distr_triple.sample()

        final_instruction = self.env.obtain_formatted_action(action_candidates_tripl[id_triple][0],
                                                             action_candidates_tripl[id_triple][1:])
        print(final_instruction)
        return final_instruction
