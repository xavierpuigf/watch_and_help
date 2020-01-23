import random
import numpy as np
from anytree import AnyNode as Node
from anytree import RenderTree
import copy
import ipdb
from profilehooks import profile

class MCTS:
    def __init__(self, env, agent_id, max_episode_length, num_simulation, max_rollout_step, c_init, c_base, seed=1):
        self.env = env
        self.discount = 0.95
        self.agent_id = agent_id
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.c_init = c_init 
        self.c_base = c_base
        self.seed = seed
        self.heuristic_dict = None
        np.random.seed(seed)
        
    def run(self, curr_root, t, heuristic_dict):

        self.heuristic_dict = heuristic_dict
        if not curr_root.is_expanded:
            curr_root = self.expand(curr_root, t)

        for explore_step in range(self.num_simulation):
            if explore_step > 0 and explore_step % 100 == 0 and self.num_simulation > 0:
                print("simulation step:", explore_step, "out of", self.num_simulation)
            curr_node = curr_root
            node_path = [curr_node]

            while curr_node.is_expanded:
                #print('Select', curr_node.id.keys())
                next_node = self.select_child(curr_node)
                if next_node is None:
                    break
                node_path.append(next_node)
                curr_node = next_node

            if next_node is None:
                continue

            leaf_node = self.expand(curr_node, t)

            value = self.rollout(leaf_node, t)
            num_actions = leaf_node.id[1][-2]
            self.backup(value*(self.discount**num_actions), node_path)

        next_root = None
        plan = []
        while curr_root.is_expanded:
            actions_taken, children_visit, next_root = self.select_next_root(curr_root)
            curr_root = next_root
            plan += actions_taken

        return next_root, plan


    def rollout(self, leaf_node, t):
        reached_terminal = False

        leaf_node_values = leaf_node.id[1]
        curr_vh_state, curr_state, goals, num_steps, actions_parent = leaf_node_values
        sum_reward = 0
        last_reward = 0

        # TODO: we should start with goals at random, or with all the goals
        # Probably not needed here since we already computed whern expanding node

        list_goals = list(range(len(goals)))
        random.shuffle(list_goals)
        for rollout_step in range(min(len(list_goals), self.max_rollout_step)):#min(self.max_rollout_step, self.max_episode_length - t)):
            goal_selected = goals[list_goals[rollout_step]]
            heuristic = self.heuristic_dict[goal_selected.split('_')[0]]
            actions = heuristic(self.agent_id, curr_state, self.env, goal_selected)
            num_steps += len(actions)
            for action_id, action in enumerate(actions):
                # Check if action can be performed
                # if action_performed:
                action_str = self.get_action_str(action)
                try:
                    next_vh_state = self.env.transition(curr_vh_state, {0: action_str})
                except:
                    ipdb.set_trace()
                next_state = next_vh_state.to_dict()
                curr_vh_state, curr_state = next_vh_state, next_state

            curr_reward = self.env.reward(0, next_state)
            delta_reward = curr_reward - last_reward# - 0.05
            delta_reward = delta_reward * self.discount**(len(actions))
            # print(curr_rewward, last_reward)
            last_reward = curr_reward
            sum_reward += delta_reward
            
    
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
        curr_state = leaf_node.id[1][1]
        if t < self.max_episode_length and not self.env.is_terminal(0, curr_state):
            expanded_leaf_node = self.initialize_children(leaf_node)
            if expanded_leaf_node is not None:
                leaf_node.is_expanded = True
                leaf_node = expanded_leaf_node
        return leaf_node


    def backup(self, value, node_list):
        for node in node_list:
            node.sum_value += value
            node.num_visited += 1
            # if value > 0:
            #     print(value, [node.id.keys() for node in node_list])
            # print(value, [node.id.keys() for node in node_list])


    def select_next_root(self, curr_root):
        children_visit = [child.num_visited for child in curr_root.children]
        children_value = [child.sum_value for child in curr_root.children]
        # print('children_ids:', children_ids)
        # print('children_visit:', children_visit)
        # print('children_value:', children_value)
        # print(list([c.id.keys() for c in curr_root.children]))
        maxIndex = np.argwhere(
            children_visit == np.max(children_visit)).flatten()
        selected_child_index = random.choice(maxIndex)
        actions = curr_root.children[selected_child_index].id[1][-1]
        return actions, children_visit, curr_root.children[selected_child_index]

    def initialize_children(self, node):
        leaf_node_values = node.id[1]
        vh_state, state, goals, steps, actions_parent = leaf_node_values


        parent_action = node.id[0]
        goal_id = leaf_node_values[2]
        
        action_space = []
        goal_incomplete = False
        for goal in goals:

            heuristic = self.heuristic_dict[goal.split('_')[0]]
            actions_heuristic = heuristic(self.agent_id, state, self.env, goal)
            next_vh_state = vh_state
            actions_str = []
            for action in actions_heuristic:
                action_str = self.get_action_str(action)
                actions_str.append(action_str)

                # TODO: this could just be computed in the heuristics?
                next_vh_state = self.env.transition(next_vh_state, {0: action_str})
            goals_remain = [goal_r for goal_r in goals if goal_r != goal]
            Node(parent=node,
                id=(goal, [next_vh_state, next_vh_state.to_dict(), goals_remain, 
                    len(actions_heuristic), actions_str]),
                 num_visited=0,
                 sum_value=0,
                 action_prior=len(goals),
                 is_expanded=False)

        return node

    def get_action_str(self, action_tuple):
        obj_args = [x for x in list(action_tuple)[1:] if x is not None]
        objects_str = ' '.join(['<{}> ({})'.format(x[0], x[1]) for x in obj_args])
        return '[{}] {}'.format(action_tuple[0], objects_str)




        
