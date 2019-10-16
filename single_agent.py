from models.single_policy import SinglePolicy

import torch
import pdb


class SingleAgent():
    def __init__(self, env, goal, agent_id, policy=None):
        self.env = env
        self.goal = goal
        self.agent_id = agent_id
        self.policy_net = policy
        if policy is not None:
            self.activation_info = policy.activation_info()
        self.beliefs = None

        self.agent_info = {
            'saved_log_probs': [],
            'indices': [],
            'action_space': [],
            'rewards': []
        }

    def reset(self):
        self.agent_info = {
            'saved_log_probs': [],
            'indices': [],
            'action_space': [],
            'rewards': []
        }



    def get_beliefs(self):
        return self.beliefs

    def update_beliefs(self, beliefs):
        self.beliefs = beliefs

    def get_observations(self):
        return self.env.get_observations()

    def update_info(self, info, reward):
        logs = info['log_probs']
        indices = info['indices']
        action_space = info['action_space']
        self.agent_info['saved_log_probs'].append(logs)
        self.agent_info['indices'].append(indices)
        self.agent_info['action_space'].append(action_space)
        self.agent_info['rewards'].append(reward)

    def get_instruction(self, observations):
        indices = []
        space = []
        log_probs = []

        distr_o1, candidates_o1 = self.policy_net.get_first_obj(observations, self)
        obj1_id = distr_o1.sample()
        object_1_selected = candidates_o1[obj1_id]
        space.append(candidates_o1)
        indices.append(obj1_id)
        log_probs.append(distr_o1.log_prob(obj1_id))

        # Decide action
        action_candidates_tripl = self.env.get_action_space(obj1=object_1_selected, structured_actions=True)
        actions_unique = list(set([x[0] for x in action_candidates_tripl]))

        # If the object is None consider the none action, that means stop
        if candidates_o1[obj1_id] is None:
            actions_unique.append(None)

        distr_a1, candidates_action = self.policy_net.get_action(actions_unique, self, obj1_id)
        action_id = distr_a1.sample()
        space.append(candidates_action)
        indices.append(action_id)
        log_probs.append(distr_a1.log_prob(action_id))

        action_selected = actions_unique[action_id]
        action_candidates_tripl = self.env.get_action_space(action=action_selected,
                                                            obj1=object_1_selected,
                                                            structured_actions=True)
        if len(action_candidates_tripl) == 1:
            id_triple = 0

        else:
            distr_triple, candidates_tripl = self.policy_net.get_second_obj(action_candidates_tripl, self)
            id_triple = distr_triple.sample()
            space.append(candidates_tripl)
            indices.append(id_triple)
            log_probs.append(distr_triple.log_prob(id_triple))

        final_instruction = self.env.obtain_formatted_action(action_candidates_tripl[id_triple][0],
                                                             action_candidates_tripl[id_triple][1:])

        dict_info = {
            'instruction': final_instruction,
            'log_probs': log_probs,
            'indices': indices,
            'action_space': space,

        }
        return dict_info
