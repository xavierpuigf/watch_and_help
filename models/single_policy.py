import torch
import pdb
import re

class SinglePolicy(torch.nn.Module):
    def __init__(self):
        super(SinglePolicy, self).__init__()
        self.saved_log_probs = []
        self.saved_indices = []
        self.rewards = []
        self.indices = []
        self.action_space = []

    def forward(self, observations):
        return []

    def update_info(self, info, reward):
        logs = info['log_probs']
        indices = info['indices']
        action_space = info['action_space']
        self.saved_log_probs.append(logs)
        self.indices.append(indices)
        self.action_space.append(action_space)
        self.rewards.append(reward)

    def bc_loss(self, program):
        num_steps = len(self.action_space)
        actions, o1, o2 = self.parse_prog(program)
        action_candidates = [x[1] for x in self.action_space]
        obj1_candidates = [[(node['class_name'], node['id']) if node is not None else None for node in x[0]] for x in self.action_space]
        if len(self.action_space[0]) > 2:
            # Assumption here is that all the actions will have the same #args
            # not necessarily true though
            obj2_candidates = [[(node['class_name'], node['id']) for node in x[1]] for x in self.action_space]
        else:
            obj2_candidates = [[None] for _ in range(num_steps)]

        # Obtain the candidates and find the matching index
        # Loss o1
        losses = []
        for it in range(num_steps):
            loss = torch.zeros([1])
            gt_action, gt_o1, gt_o2 = actions[it], o1[it], o2[it]
            index_action = [it for it,x in enumerate(action_candidates[it]) if x.upper() == gt_action]
            index_o1 = [it for it, x in enumerate(obj1_candidates[it]) if x == gt_o1]
            index_o2 = [it for it, x in enumerate(obj2_candidates[it]) if x == gt_o2]
            index_action = index_action[0] if len(index_action) > 0 else None
            index_o1 = index_o1[0] if len(index_o1) > 0 else None
            index_o2 = index_o2[0] if len(index_o2) > 0 else None
            #pdb.set_trace()
            if len(action_candidates[it]) > 1 and index_action is not None:
                loss += -self.saved_log_probs[it][1] # action
            if len(obj1_candidates[it]) > 1 and index_o1 is not None:
                loss += -self.saved_log_probs[it][0] # object1
            if len(obj2_candidates[it]) > 1 and index_o2 is not None:
                loss += -self.saved_log_probs[it][2] # object2
            losses.append(loss)

        losses = torch.cat(losses)
        return losses.sum()

    def pg_loss(self, labels):
        num_step = len(self.rewards)
        policy_loss = []
        for it in num_step:
            reward = self.rewards[it]
            log_prob = self.saved_log_probs[it]
            policy_loss.append(-log_prob*reward)
        return torch.cat(policy_loss).sum()

    def parse_prog(self, prog):
        program = []
        actions = []
        o1 = []
        o2 = []
        for progstring in prog:
            params = []

            patt_action = r'^\[(\w+)\]'
            patt_params = r'\<(.+?)\>\s*\((.+?)\)'

            action_match = re.search(patt_action, progstring.strip())
            action_string = action_match.group(1).upper()


            param_match = re.search(patt_params, action_match.string[action_match.end(1):])
            while param_match:
                params.append((param_match.group(1), int(param_match.group(2))))
                param_match = re.search(patt_params, param_match.string[param_match.end(2):])

            program.append((action_string, params))
            actions.append(action_string)
            if len(params) > 0:
                o1.append(params[0])
            else:
                o1.append(None)
            if len(params) > 1:
                o2.append(params[1])
            else:
                o2.append(None)

        return actions, o1, o2