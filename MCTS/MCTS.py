import random
import numpy as np
from anytree import AnyNode as Node
from anytree import RenderTree
import copy


class MCTS:
    def __init__(self, env, max_episode_length, num_simulation, max_rollout_step, c_init, c_base, seed=1):
        self.env = env
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.c_init = c_init 
        self.c_base = c_base
        self.seed = seed
        np.random.seed(seed)
        

    def run(self, curr_root, t, obj_heuristic):
        self.obj_heuristic = obj_heuristic
        if not curr_root.is_expanded:
            curr_root = self.expand(curr_root, t)
        for explore_step in range(self.num_simulation):
            if explore_step % 100 == 0 and self.num_simulation > 0:
                print("simulation step:", explore_step, "out of", self.num_simulation)
            curr_node = curr_root
            node_path = [curr_node]

            while curr_node.is_expanded:
                next_node = self.select_child(curr_node)
                # print(next_node)
                if next_node is None: break
                node_path.append(next_node)
                curr_node = next_node
            if next_node is None: continue
            leaf_node = self.expand(curr_node, t)
            value = self.rollout(leaf_node, t)
            # print(value)
            self.backup(value, node_path)
            
        next_root = None
        plan = []
        while curr_root.is_expanded:
            action_taken, children_visit, next_root = self.select_next_root(curr_root)
            curr_root = next_root
            plan.append(action_taken)
        return next_root, plan


    def rollout(self, leaf_node, t):
        reached_terminal = False
        curr_vh_state, curr_state = list(leaf_node.id.values())[0][0], list(leaf_node.id.values())[0][1]
        sum_reward = 0
        last_reward = 0
        for rollout_step in range(self.max_rollout_step):#min(self.max_rollout_step, self.max_episode_length - t)):
            action_space = []
            for obj in self.obj_heuristic:
                action_space += self.env.get_action_space(curr_vh_state, obj1=obj)
            # action_space = self.env.get_action_space(curr_vh_state, obj1=self.obj_heuristic)
            rollout_policy = lambda state: random.choice(action_space)
            action = rollout_policy(curr_state)
            # print('rollout:', [e for e in curr_state['edges'] if 2038 in e.values()])
            # if action == '[walk] <bench> (190)':
            #     print('here:', self.env.is_terminal(0, curr_state), self.env.reward(0, curr_state))
            #     input('press any key to continue...')
            # print(curr_state)
            # print(self.env.is_terminal(0, curr_state))
            if self.env.is_terminal(0, curr_state):# or t + rollout_step + 1 >= self.max_episode_length:
                sum_reward = self.env.reward(0, curr_state)
                reached_terminal = True
                break
            next_vh_state = self.env.transition(curr_vh_state, {0: action})
            next_state = next_vh_state.to_dict()
            curr_rewward = self.env.reward(0, next_state)
            delta_reward = curr_rewward - last_reward# - 0.05
            # print(curr_rewward, last_reward)
            last_reward = curr_rewward
            sum_reward += delta_reward
            curr_vh_state, curr_state = next_vh_state, next_state
        # print(sum_reward, reached_terminal)
        return sum_reward


    def calculate_score(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        action_prior = child.action_prior

        if self_visit_count == 0:
            u_score = 1e6 #np.inf
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) /
                                      self.c_base) + self.c_init
            u_score = exploration_rate * action_prior * np.sqrt(
                parent_visit_count) / float(1 + self_visit_count)
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score


    def select_child(self, curr_node):
        scores = [
            self.calculate_score(curr_node, child)
            for child in curr_node.children
        ]
        if len(scores) == 0: return None
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child
        

    def get_action_prior(self, action_space):
        action_space_size = len(action_space)
        action_prior = {
            action: 1.0 / action_space_size
            for action in action_space
        }
        return action_prior


    def expand(self, leaf_node, t):
        curr_state = list(leaf_node.id.values())[0][1]
        if t < self.max_episode_length and not self.env.is_terminal(0, curr_state):
            leaf_node.is_expanded = True
            leaf_node = self.initialize_children(leaf_node)
        return leaf_node


    def backup(self, value, node_list):
        for node in node_list:
            node.sum_value += value
            node.num_visited += 1
            # if value > 0:
            #     print(value, [node.id.keys() for node in node_list])
            # print(value, [node.id.keys() for node in node_list])


    def select_next_root(self, curr_root):
        # children_ids = [list(child.id.values())[0] for child in curr_root.children]
        children_visit = [child.num_visited for child in curr_root.children]
        children_value = [child.sum_value for child in curr_root.children]
        # print('children_ids:', children_ids)
        print('children_visit:', children_visit)
        print('children_value:', children_value)
        print(list([c.id.keys() for c in curr_root.children]))
        maxIndex = np.argwhere(
            children_visit == np.max(children_visit)).flatten()
        selected_child_index = random.choice(maxIndex)
        action = list(curr_root.children[selected_child_index].id.keys())[0]
        return action, children_visit, curr_root.children[selected_child_index]


    def initialize_children(self, node):
        vh_state, state = list(node.id.values())[0][0], list(node.id.values())[0][1]
        # print(state['nodes'])
        # print('edges about character', [x for x in state['edges'] if x['from_id'] == 65])# and x['relation_type'] in ['INSIDE', 'CLOSE']])
        # print('edges about cup', [x for x in state['edges'] if x['from_id'] == 2009])
        action_space = []
        for obj in self.obj_heuristic:
            action_space += self.env.get_action_space(vh_state, obj1=obj)
        # print('initialize_children:', self.env.get_action_space(vh_state))
        # print('initialize_children:', action_space)
        # print(self.env.observable_object_ids_n[0])
        # input('press any key to continue....')
        init_action_prior = self.get_action_prior(action_space) # TODO: ction space decomposition -- o1, action, o2
        for action in action_space:
            # print(action, self.env.pomdp)
            next_vh_state = self.env.transition(vh_state, {0: action})
            Node(parent=node,
                 id={action: [next_vh_state, next_vh_state.to_dict()]},
                 num_visited=0,
                 sum_value=0,
                 action_prior=init_action_prior[action],
                 is_expanded=False)
        return node


    def rollout_heuristic(self):
        pass
        