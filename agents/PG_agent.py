import numpy as np
from pathlib import Path
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing

from .base_agent import BaseAgent
import vh_graph
from vh_graph.envs import belief
from vh_graph.envs.vh_env import VhGraphEnv

import torch
from torch import distributions
import utils
import envdataset
from models.single_policy import SinglePolicy


def sample_instruction(env, dataset, state, action_logits, o1_logits, o2_logits, clip_value, pick_max=False):

    instruction = None
    object_ids = state[1]
    object_names = state[0]

    distr_object1 = distributions.categorical.Categorical(logits=o1_logits)
    if pick_max:
        obj1_id_model = torch.argmax(o1_logits, -1)
    else:
        candidate_ids = torch.where(o1_logits > clip_value)[-1]
        obj1_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()
        # Uniform sampling

    prob_1d = distr_object1.log_prob(obj1_id_model)
    obj1_id = state[1][0, 0, obj1_id_model].item()

    if obj1_id < 0:
        # TODO: sometimes this none could correspond to standing up, we will need to solve this
        instruction = 'stop'
        action_candidate_ids = [dataset.action_dict.get_id('stop')]

    else:
        try:
            object1_node = [x for x in env.vh_state.to_dict()['nodes'] if x['id'] == obj1_id][0]
        except:
            import pdb
            pdb.set_trace()
        # Given this object, get the action candidates
        action_candidates_tripl = env.get_action_space(obj1=object1_node, structured_actions=True)

        action_candidate_ids = [dataset.action_dict.get_id(x[0]) for x in action_candidates_tripl if
                                x[0].lower() in ['walk', 'open']]

    mask = torch.zeros(action_logits[0, 0].shape).cuda()
    mask[action_candidate_ids] = 1.

    action_logits_masked = action_logits * mask + clip_value * (1 - mask)
    distr_action1 = distributions.categorical.Categorical(logits=action_logits_masked)
    if pick_max:
        action_id_model = action_logits_masked.argmax(-1)
    else:
        candidate_ids = torch.where(action_logits_masked > clip_value)[-1]
        action_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()

    prob_action = distr_action1.log_prob(action_id_model)

    # Given action and object, get the last candidates
    if obj1_id < 0:
        obj2_id_cands = torch.tensor([dataset.node_none[1]])
    else:
        action_selected = dataset.action_dict.get_el(action_id_model.item())
        if action_selected.upper() == 'OTHER':
            pdb.set_trace()
        triple_candidates = env.get_action_space(action=action_selected,
                                                      obj1=object1_node,
                                                      structured_actions=True)

        obj2_id_cands = torch.tensor([dataset.node_none[1] if len(x) < 3 else x[2]['id'] for x in triple_candidates])
        if len(triple_candidates) == 0:
            pdb.set_trace()
    try:

        mask_o2 = (state[1] == obj2_id_cands[None, None]).float().cuda()
    except:
        import ipdb
        ipdb.set_trace()
    o2_logits_masked = (o2_logits * mask_o2) + (1 - mask_o2) * clip_value
    distr_object2 = distributions.categorical.Categorical(logits=o2_logits_masked)

    if pick_max:
        obj2_id_model = torch.argmax(o2_logits_masked, -1)
    else:
        candidate_ids = torch.where(o2_logits_masked > clip_value)[-1]
        obj2_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()

    prob_2d = distr_object2.log_prob(obj2_id_model)
    # if instruction is None:
    try:
        instruction = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                   [action_id_model, obj1_id_model, obj2_id_model])
    except:
        pdb.set_trace()
    instruction = instruction[0]
    return instruction, (prob_action, prob_1d, prob_2d)



class PG_agent(BaseAgent):
    """
    PG agent
    """

    def __init__(self, env, max_episode_length, num_simulation, max_rollout_steps):
        self.env = env
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps

        self.clip_value = -1e9

        args = utils.read_args()
        args.invert_edge = True

        self.dataset = envdataset.EnvDataset(args, process_progs=False)

        policy_net = SinglePolicy(self.dataset).cuda()
        policy_net = torch.nn.DataParallel(policy_net)

        weights = '../../vh_multiagent_models/logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_' \
                  'training_mode.pg_invertedge.True/offp.False_eps.0.2_gamma.0.7/2019-11-20_15.20.44.417376/' \
                  'chkpt/chkpt_249.pt'

        state_dict = torch.load(weights)
        policy_net.load_state_dict(state_dict['model_params'])

        self.policy_net = policy_net
        self.previous_belief_graph = None

        nodes, _, self.ids_used = self.dataset.process_graph(env.vh_state.to_dict())
        self.class_names, self.object_ids, _, self.mask_nodes, _ = nodes


    def obtain_logits_from_observations(self, curr_state, visible_ids, goal_str):

        id_char = self.dataset.object_dict.get_id('character')
        nodes, edges, _ = self.dataset.process_graph(curr_state, self.ids_used, visible_ids)
        _, _, visible_mask, _, state_nodes = nodes
        edge_bin, edge_types, mask_edges = edges
        graph_data = self.dataset.join_timesteps(self.class_names, self.object_ids, [state_nodes],
                                                        [edge_bin], [edge_types], [visible_mask],
                                                        self.mask_nodes, [mask_edges])

        goal_info = self.dataset.prepare_goal('findnode_{}'.format(goal_str), self.ids_used, self.class_names)
        graph_data = [torch.tensor(x).unsqueeze(0) for x in graph_data]
        goal_info = [torch.tensor(x).unsqueeze(0) for x in list(goal_info)]

        output = self.policy_net(graph_data, goal_info, id_char)
        action_logits, o1_logits, o2_logits, _ = output
        mask_character = (graph_data[0] != id_char).float().cuda()
        o1_logits = o1_logits * mask_character + (1 - mask_character) * self.clip_value
        return graph_data, action_logits, o1_logits, o2_logits

    def one_step_rollout(self, env, curr_state, visible_ids, goal_id):
        graph_data, action_logits, o1_logits, o2_logits = self.obtain_logits_from_observations(
            curr_state, visible_ids, goal_id)
        instruction, logits = sample_instruction(env, self.dataset, graph_data,
                                                 action_logits, o1_logits, o2_logits, clip_value=self.clip_value)
        instr = list(zip(*instruction))[0]
        str_instruction = utils.pretty_instr(instr)
        return str_instruction, logits

    def get_plan(self, env, nb_steps, goal_id):
        init_state = env.state

        if env.is_terminal(0, init_state):
            return

        iter = 0
        terminal = False
        plan = []
        print(terminal, iter, nb_steps)
        while iter < nb_steps and not terminal:
            curr_vh_state = self.sim_env.vh_state.to_dict()
            visible_ids = self.sim_env.observable_object_ids_n[0]
            instr, log = self.one_step_rollout(env, curr_vh_state, visible_ids, goal_id)

            iter += 1
            if 'stop' not in instr:
                plan.append(instr)
                print(instr)
                _ = self.sim_env.step({0: instr})
            else:
                terminal = True

        return plan


    def rollout(self, graph, task_goal):
        goal_id = int(task_goal[0].split('_')[-1])
        nb_steps = 0
        _ = self.env.reset(graph, task_goal)
        done = False      
        self.env.to_pomdp()
        gt_state = self.env.vh_state.to_dict()
        self.belief = belief.Belief(gt_state)
        self.sample_belief(self.env.get_observations(0))
        self.sim_env.reset(self.previous_belief_graph, task_goal)
        self.sim_env.to_pomdp()

        root_action = None
        root_node = None
        # print(self.sim_env.pomdp)
        history = {'belief': [], 'plan': [], 'action': [], 'sampled_state': []}
        while not done and nb_steps < self.max_episode_length:
            plan = self.get_plan(self.sim_env, self.max_episode_length - nb_steps, goal_id)
            root_action = None
            action = plan[0].strip()
            print(action, 'HISTORY', history['action'])

            if action in self.env.get_action_space():
                history['belief'].append(copy.deepcopy(self.belief.edge_belief))
                history['plan'].append(plan)
                history['action'].append(action)
                history['sampled_state'].append(self.sim_env.vh_state.to_dict())

                reward, state, infos = self.env.step({0: action})
                done = abs(reward[0] - 1.0) < 1e-6
                _, _, _ = self.sim_env.step({0: action})
                nb_steps += 1
                print(nb_steps, action, reward)
                self.sample_belief(self.env.get_observations(0))
                self.sim_env.reset(self.previous_belief_graph, task_goal)
            else:
                import pdb
                pdb.set_trace()
                break
            

            state = self.env.vh_state.to_dict()
            sim_state = self.sim_env.vh_state.to_dict()
            self.sim_env.to_pomdp()
            id_agent = [x['id'] for x in state['nodes'] if x['class_name'] == 'character'][0]
            print('real state:', [e for e in state['edges'] if goal_id in e.values()])
            print('real state:', [e for e in state['edges'] if id_agent in e.values()])

            print('sim state:', [e for e in sim_state['edges'] if goal_id in e.values()])# and e['relation_type'] == 'INSIDE'])
            print('sim state:', [e for e in sim_state['edges'] if e['from_id'] == 229])
            # print([e for e in sim_state['edges'] if 117 in e.values() and e['relation_type'] == 'INSIDE'])
            print('sim state:', [e for e in sim_state['edges'] if id_agent in e.values()])
            input('press any key to continue...')

            pickle.dump(history, open('logdir/history_pg.pik', 'wb'))

            # print('action_space:', self.env.get_action_space(obj1=['cup', 'cupboard', 'dining_room']))
