from models.single_policy import SinglePolicy

import torch
from torch import distributions
import pdb
from tqdm import tqdm
import vh_graph
from vh_graph.envs import belief
from vh_graph.envs.vh_env import VhGraphEnv
import gym
import json
import envdataset
import utils_viz
import utils
import random
import numpy as np

class SingleAgent():
    def __init__(self, env, goal, agent_id, dataset=None, policy=None, use_belief=False):
        self.env = env
        self.goal = goal
        self.agent_id = agent_id
        self.policy_net = policy
        self.dataset = dataset
        self.clip_value = -1e9
        self.use_belief = use_belief

        self.max_steps = 10
        #if policy is not None:
        #    self.activation_info = policy.activation_info()
        if self.use_belief:
            self.beliefs = None
        self.finished = False

        self.agent_info = {
            'saved_log_probs': [],
            'indices': [],
            'action_space': [],
            'rewards': []
        }

        if dataset is not None:
            gt_state = env.vh_state.to_dict()
            self.node_id_char = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]
            # All the nodes
            nodes, _, ids_used = dataset.process_graph(gt_state)
            class_names, object_ids, _, mask_nodes, _ = nodes
            self.nodes = nodes
            self.ids_used = ids_used
            self.class_names = class_names
            self.object_ids = object_ids
            self.mask_nodes = mask_nodes

            if self.use_belief:
                self.belief = belief.Belief(gt_state)

        if self.use_belief:
            self.previous_belief_graph = None
            self.belief_state = None
            self.belief_sim = VhGraphEnv()
            self.belief_sim.pomdp = True

    def sample_belief(self, obs_graph):
        self.belief.update_from_gt_graph(obs_graph)
        if self.previous_belief_graph is None:
            self.belief.reset_belief()
            new_graph = self.belief.sample_from_belief()
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph
        else:
            new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
            self.previous_belief_graph = new_graph
        self.belief_sim.reset_graph(self.previous_belief_graph)

    def get_observations(self):
        return self.env.get_observations()


    def get_top_instruction(self, dataset, state, action_logits, o1_logits, o2_logits):
        pred_action = torch.argmax(action_logits, -1)
        pred_o1 = torch.argmax(o1_logits, -1)
        pred_o2 = torch.argmax(o2_logits, -1)
        object_ids = state[1]
        object_names = state[0]
        pred_instr = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                  [pred_action, pred_o1, pred_o2])
        return pred_instr[0]

    def sample_instruction(self, dataset, state, action_logits, o1_logits, o2_logits, pick_max=False, offp_action=None, eps=0.):
        be_greedy = np.random.uniform() > eps

        instruction = None
        object_ids = state[1]
        object_names = state[0]

        distr_object1 = distributions.categorical.Categorical(logits=o1_logits)
        if offp_action is not None:
            object_id = offp_action[1][1] if offp_action[1] is not None else -1
            obj1_id_model = torch.where(object_ids[0,0] == object_id)[0][None, :].cuda()

        else:
            if pick_max:
                obj1_id_model = torch.argmax(o1_logits, -1)
            else:
                if be_greedy:
                    obj1_id_model = distr_object1.sample()
                else:
                    candidate_ids = torch.where(o1_logits > self.clip_value)[-1]
                    obj1_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()
                    # Uniform sampling

        prob_1d = distr_object1.log_prob(obj1_id_model)
        obj1_id = state[1][0,0,obj1_id_model].item()

        if obj1_id < 0:
            # TODO: sometimes this none could correspond to standing up, we will need to solve this
            instruction = 'stop'
            action_candidate_ids = [dataset.action_dict.get_id('stop')]

        else:
            try:
                object1_node = [x for x in self.env.vh_state.to_dict()['nodes'] if x['id'] == obj1_id][0]
            except:
                pdb.set_trace()
            # Given this object, get the action candidates
            action_candidates_tripl = self.env.get_action_space(obj1=object1_node, structured_actions=True)

            action_candidate_ids = [dataset.action_dict.get_id(x[0]) for x in action_candidates_tripl if x[0].lower() != 'grab']

        mask = torch.zeros(action_logits[0,0].shape).cuda()
        mask[action_candidate_ids] = 1.

        action_logits_masked = action_logits * mask + self.clip_value * (1-mask)
        distr_action1 = distributions.categorical.Categorical(logits=action_logits_masked)

        if offp_action:
            action_name = offp_action[0]
            action_id_model = torch.tensor([[dataset.action_dict.get_id(action_name)]]).cuda()
        else:
            if pick_max:
                action_id_model = action_logits_masked.argmax(-1)
            else:
                if be_greedy:
                    action_id_model = distr_action1.sample()
                else:
                    candidate_ids = torch.where(action_logits_masked > self.clip_value)[-1]
                    action_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()

        prob_action = distr_action1.log_prob(action_id_model)

        # Given action and object, get the last candidates
        if obj1_id < 0:
            obj2_id_cands = torch.tensor([dataset.node_none[1]])
        else:
            action_selected = dataset.action_dict.get_el(action_id_model.item())
            if action_selected.upper() == 'OTHER':
                pdb.set_trace()
            triple_candidates = self.env.get_action_space(action=action_selected,
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
        o2_logits_masked = (o2_logits * mask_o2) + (1-mask_o2)*self.clip_value
        distr_object2 = distributions.categorical.Categorical(logits=o2_logits_masked)

        if pick_max:
            obj2_id_model = torch.argmax(o2_logits_masked, -1)
        else:
            if be_greedy:
                obj2_id_model = distr_object2.sample()
            else:
                candidate_ids = torch.where(o2_logits_masked > self.clip_value)[-1]
                obj2_id_model = torch.tensor([[candidate_ids[np.random.randint(candidate_ids.shape[0])]]]).cuda()

        prob_2d = distr_object2.log_prob(obj2_id_model)
        #if instruction is None:
        try:
            instruction = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                       [action_id_model, obj1_id_model, obj2_id_model])
        except:
            pdb.set_trace()
        instruction = instruction[0]
        return instruction, (prob_action, prob_1d, prob_2d)


    def obtain_logits_from_observations(self, curr_state, visible_ids, goal_str):

        id_char = self.dataset.object_dict.get_id('character')
        nodes, edges, _ = self.dataset.process_graph(curr_state, self.ids_used, visible_ids)
        _, _, visible_mask, _, state_nodes = nodes
        edge_bin, edge_types, mask_edges = edges
        graph_data = self.dataset.join_timesteps(self.class_names, self.object_ids, [state_nodes],
                                                        [edge_bin], [edge_types], [visible_mask],
                                                        self.mask_nodes, [mask_edges])

        goal_info = self.dataset.prepare_goal(goal_str, self.ids_used, self.class_names)
        graph_data = [torch.tensor(x).unsqueeze(0) for x in graph_data]
        goal_info = [torch.tensor(x).unsqueeze(0) for x in list(goal_info)]

        output = self.policy_net(graph_data, goal_info, id_char)
        action_logits, o1_logits, o2_logits, _ = output
        mask_character = (graph_data[0] != id_char).float().cuda()
        o1_logits = o1_logits * mask_character + (1 - mask_character) * self.clip_value
        return graph_data, action_logits, o1_logits, o2_logits

    def one_step_rollout(self, goal_string, pomdp, remove_edges=False, offp_action=None, use_belief=False, eps=0.):
        if use_belief:
            observations = self.get_observations()
            self.sample_belief(observations)
            curr_state = self.belief_sim.vh_state.to_dict()
            visible_ids = self.belief_sim.observable_object_ids_n[0]
        else:
            if pomdp:
                curr_state = self.get_observations()
                visible_ids = None
            else:
                curr_state = self.env.vh_state.to_dict()
                visible_ids = self.env.observable_object_ids_n[0]

        if remove_edges:
            curr_state['edges'] = [x for x in curr_state['edges'] if x['relation_type'] != 'CLOSE']

        graph_data, action_logits, o1_logits, o2_logits = self.obtain_logits_from_observations(
            curr_state, visible_ids, goal_string)
        instruction, logits = self.sample_instruction(self.dataset, graph_data,
                                                      action_logits, o1_logits, o2_logits, offp_action=offp_action, eps=eps)
        instr = list(zip(*instruction))[0]
        str_instruction = utils.pretty_instr(instr)

        if 'stop' in str_instruction:
            resp = None
        else:
            if use_belief:
                self.belief_sim.step({0: str_instruction})
                self.belief.sampled_graph = self.belief_sim.vh_state.to_dict()
            resp = self.env.step({0: str_instruction})
        # Measure reward
        # TODO: this should be done in the env
        goal_achieved = self.goal_achieved(goal_string)
        if 'stop' in str_instruction:
            if goal_achieved:
                reward = 1
            else:
                reward = 0
        else:
            if goal_achieved:
                reward = 1
            else:
                reward = 0
        


        return resp, str_instruction, logits, reward, (o1_logits, graph_data[1])

    def goal_achieved(self, goal_string):
        goal_id = int(goal_string.split('_')[-1])

        # Similar objects, we will consider a success all objects of the same class and in the same location
        curr_graph = self.env.vh_state.to_dict()
        node_interest = [x for x in curr_graph['nodes'] if x['id'] == goal_id][0]
        edges_interest = [(edge['to_id'], edge['relation_type']) for edge in curr_graph['edges'] if edge['from_id'] == node_interest['id']]
        edges_interest += [(edge['from_id'], edge['relation_type']) for edge in curr_graph['edges'] if edge['to_id'] == node_interest['id']]

        edges_interest = set(edges_interest)

        nodes_with_class_name = [x['id'] for x in curr_graph['nodes'] if x['class_name'] == node_interest['class_name'] and x['id'] != node_interest['id']]
        nodes_same_specs = [node_interest['id']]
        for node_other in nodes_with_class_name:
            edges_other = [(edge['from_id'], edge['relation_type']) for edge in curr_graph['edges'] if
                            edge['to_id'] == node_other]
            edges_other += [(edge['to_id'], edge['relation_type']) for edge in curr_graph['edges'] if
                           edge['from_id'] == node_other]
            edges_other = set(edges_other)
            if edges_other == edges_interest:
                nodes_same_specs.append(node_other)


        for node_goal in nodes_same_specs:
            edges_goal = [x for x in curr_graph['edges']
                          if x['relation_type'] == 'CLOSE' and x['from_id'] == node_goal and x['to_id'] == self.node_id_char]
            edge_found = len(edges_goal) > 0
            if node_goal in self.env.observable_object_ids_n[0] and edge_found:
                return True
        return False


    def rollout(self, goal_str, pomdp, actions_off_policy=None, eps=0., terminate_at_goal=False):
        max_instr = 0
        instr = ''
        instructions = []
        logits = []
        rewards = []
        offp_actions = [None]*self.max_steps if actions_off_policy is None else list(zip(*actions_off_policy))
        o1l_l = []
        terminate = False
        while max_instr < self.max_steps and not terminate:
            _, instr, log, r, o1l = self.one_step_rollout(goal_str, pomdp,
                                                     remove_edges=False,
                                                     offp_action=offp_actions[max_instr], eps=eps)
            terminate = 'stop' in instr or (terminate_at_goal and r > 0.)
            max_instr += 1
            instructions.append(instr)
            logits.append(log)
            rewards.append(r)
            o1l_l.append(o1l)
        return instructions, logits, rewards, o1l_l

def dataset_agent():
    args = utils.read_args()
    success_per_length = {0: [0,0], 1: [0,0], 2: [0,0], 3: [0,0], 4: [0,0], 5: [0,0], 6: [0,0]}
    if args.pomdp:
        #weights = 'logdir/dataset_folder.dataset_toy3_pomdp.True_graphsteps.3_training_mode.bc/2019-11-07_12.40.50.558146/chkpt/chkpt_49.pt'
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.True_graphsteps.3_training_mode.bc/2019-11-12_21.33.22.040142/chkpt/chkpt_149.pt'
        weights = 'logdir/dataset_folder.dataset_toy4_pomdp.True_graphsteps.3_training_mode.bc/2019-11-13_19.24.37.715547/chkpt/chkpt_149.pt'
    else:
        #weights = 'logdir/dataset_folder.dataset_toy3_pomdp.False_graphsteps.3_training_mode.bc/2019-11-07_12.38.44.852796/chkpt/chkpt_49.pt'
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.bc/2019-11-12_21.33.57.388801/chkpt/chkpt_149.pt'
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.bc/2019-11-13_19.25.08.723550/chkpt/chkpt_149.pt'
        weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.pg/offp.False_eps.0.2_gamma.0.7/2019-11-14_00.45.32.080556/chkpt/chkpt_149.pt'

    # 'logdir/pomdp.True_graphsteps.3/2019-10-30_17.35.51.435717/chkpt/chkpt_61.pt'

    time_agent = utils.AvgMetrics(['time_reset', 'time_get_graph', 'time_model',
                                   'time_observations', 'time_sample', 'time_goal', 'time_step',
                                   'time_agent_creation', 'time_total'], ':.5f')
    print('Loaded')
    # Set up the policy
    curr_env = gym.make('vh_graph-v0')
    args.max_steps = 1
    args.interactive = True
    args.dataset_folder = 'dataset_toy4'
    helper = utils.Helper(args)

    dataset = envdataset.EnvDataset(args, split='test')
    policy_net = SinglePolicy(dataset).cuda()
    policy_net = torch.nn.DataParallel(policy_net)
    policy_net.eval()

    if weights is not None:
        print('Loading weights')

        state_dict = torch.load(weights)
        policy_net.load_state_dict(state_dict['model_params'])

    import time
    print('loaded')
    final_list = []
    success, cont_episodes, avg_len = 0, 0, 0
    with torch.no_grad():
        for problem in tqdm(dataset.problems_dataset):
            time_init = time.time()
            path_init_env = problem['graph_file']
            goal_str = problem['goal']
            print('Goal: {}'.format(goal_str))
            print(path_init_env)
            goal_name = '(facing living_room[1] living_room[1])'
            curr_env.reset(path_init_env, {0: goal_name})
            curr_env.to_pomdp()
            time_reset = time.time()

            single_agent = SingleAgent(curr_env, goal_name, 0, dataset, policy_net)

            time_agent_creation = time.time()
            gt_state = single_agent.env.vh_state.to_dict()
            node_id_char = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]
            # All the nodes
            nodes, _, ids_used = dataset.process_graph(gt_state)
            class_names, object_ids, _, mask_nodes, _ = nodes

            time_get_graph = time.time()
            finished = False
            cont = 0
            curr_success = False
            if single_agent.goal_achieved(goal_str):
                continue

            time_goal = 0.
            time_sample = 0.
            time_obs = 0.
            time_model = 0.
            time_step = 0.
            while cont < 10 and not finished:
                time1 = time.time()
                if args.pomdp:
                    curr_state = single_agent.get_observations()
                    visible_ids = None
                else:
                    curr_state = single_agent.env.vh_state.to_dict()
                    visible_ids = single_agent.env.observable_object_ids_n[0]

                time_observations = time.time()
                #if cont == 0:
                #    curr_state['edges'] = [x for x in curr_state['edges'] if x['relation_type'] != 'CLOSE']

                graph_data, action_logits, o1_logits, o2_logits = single_agent.obtain_logits_from_observations(
                    curr_state, visible_ids, goal_str)

                time_mod = time.time()
                instruction, _ = single_agent.sample_instruction(dataset, graph_data, action_logits, o1_logits, o2_logits, pick_max=True)
                instr = list(zip(*instruction))[0]
                str_instruction = utils.pretty_instr(instr)
                #print(str_instruction)

                time_sample_instruction = time.time()
                goal_achieved = single_agent.goal_achieved(goal_str)
                time_go = time.time()
                if goal_achieved:
                    success += 1
                    curr_success = True
                    finished = True

                else:
                    if str_instruction.strip() == '[stop]':
                        finished = True
                    else:
                        single_agent.env.step({0: str_instruction})
                    cont += 1
                time_final = time.time()

                time_step += time_final - time_go
                time_goal += time_go - time_sample_instruction
                time_sample += time_sample_instruction - time_mod
                time_obs += time_observations - time1
                time_model += time_mod - time_observations
            time_end = time.time() - time_init
            time_agent.update({'time_reset': time_reset - time_init,
                               'time_agent_creation': time_agent_creation - time_reset,
                               'time_get_graph': time_get_graph - time_agent_creation,
                               'time_model': time_model,
                               'time_observations': time_obs,
                               'time_sample': time_sample,
                               'time_goal': time_goal,
                               'time_step': time_step,
                               'time_total': time_end})
            print(time_agent)
            goal_id = int(goal_str.split('_')[-1])


            final_list.append((path_init_env, goal_id, curr_success))
            cont_episodes += 1
            avg_len += cont

            c = success_per_length[len(problem['program'])]
            increment = 1 if curr_success else 0
            success_per_length[len(problem['program'])] = [c[0] + increment, c[1]+1]
            print(success, cont_episodes, cont, curr_success, len(problem['program']))
            print('---')
    # with open('output_{}.json'.format(args.pomdp), 'w+') as f:
    #     f.write(json.dumps(final_list))
    print(success_per_length)
    # 363 fomdp vs 296 pomdp
    # FOMDP: {0: [0, 0], 1: [22, 0], 2: [168, 0], 3: [124, 0], 4: [13, 0], 5: [0, 0], 6: [0, 0]}  327 466
    # POMDP: {0: [0, 0], 1: [24, 0], 2: [153, 0], 3: [176, 0], 4: [12, 0], 5: [0, 0], 6: [0, 0]}  365 466
# def policy_gradient():

def interactive_agent():
    path_init_env = 'dataset_toy3/init_envs/TrimmedTestScene6_graph_42.json'
    goal_name = '(facing living_room[1] living_room[1])'
    weights = 'logdir/dataset_folder.dataset_toy3_pomdp.False_graphsteps.3/2019-11-06_09.14.31.202175/chkpt/chkpt_31.pt'
    # 'logdir/pomdp.True_graphsteps.3/2019-10-30_17.35.51.435717/chkpt/chkpt_61.pt'

    # Set up the policy
    curr_env = gym.make('vh_graph-v0')
    curr_env.reset(path_init_env, {0: goal_name})
    curr_env.to_pomdp()
    args = utils.read_args()
    args.max_steps = 1
    args.interactive = True
    helper = utils.Helper(args)
    dataset_interactive = envdataset.EnvDataset(args, process_progs=False)

    print('Starting model...')
    policy_net = SinglePolicy(dataset_interactive).cuda()
    policy_net = torch.nn.DataParallel(policy_net)
    policy_net.eval()
    single_agent = SingleAgent(curr_env, goal_name, 0, dataset_interactive, policy_net)

    # Starting the scene
    curr_state = single_agent.get_observations()
    gt_state = single_agent.env.vh_state.to_dict()

    # All the nodes
    nodes, _, ids_used = dataset_interactive.process_graph(gt_state)
    class_names, object_ids, _, mask_nodes, _ = nodes


    utils_viz.print_graph(gt_state, 'gt_graph.gv')
    if weights is not None:
        print('Loading weights')
        state_dict = torch.load(weights)
        policy_net.load_state_dict(state_dict['model_params'])
    id_str = input('Id of object to find...')
    id_obj = int(id_str)
    goal_str = 'findnode_{}'.format(id_obj)
    id_char = dataset_interactive.object_dict.get_id('character')

    while True:
        if args.pomdp:
            curr_state = single_agent.get_observations()
            visible_ids = None
        else:
            curr_state = single_agent.env.vh_state.to_dict()
            visible_ids = single_agent.env.observable_object_ids_n[0]

        # Process data
        graph_data, action_logits, o1_logits, o2_logits = single_agent.obtain_logits_from_observations(
            dataset_interactive, curr_state, ids_used, visible_ids, class_names, object_ids, mask_nodes, goal_str)

        instruction, _ = single_agent.sample_instruction(dataset_interactive, graph_data, action_logits, o1_logits, o2_logits)

        #instruction, _ = single_agent.sample_instruction(dataset_interactive, graph_data, action_logits, o1_logits,
        #                                                 o2_logits)

        instr = list(zip(*instruction))[0]
        str_instruction = utils.pretty_instr(instr)
        if str_instruction.strip() == '[stop]':
            print('Episode finished')
        else:
            single_agent.env.step({0: str_instruction})
        pdb.set_trace()


def train():
    args = utils.read_args()
    args.training_mode = 'pg'
    args.num_epochs = 250
    args.max_steps = 1
    args.batch_size = 1
    args.envstop = True
    args.eps_greedy = 0.2
    args.gamma = 0.7
    args.lr = 1e-4
    args.dataset_folder = 'dataset_toy4'

    helper = utils.Helper(args)
    print('Creating dataset')
    dataset = envdataset.EnvDataset(args, split='train')
    print('done')
    curr_envs = [gym.make('vh_graph-v0') for _ in range(args.batch_size)]
    num_elems = len(dataset.problems_dataset)
    shuffle_indices = list(range(num_elems))
    num_iter = num_elems // args.batch_size

    # Set up the policy
    policy_net = SinglePolicy(dataset)
    policy_net.cuda()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    policy_net = torch.nn.DataParallel(policy_net)
    weights = None
    if args.pomdp:
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.True_graphsteps.3_training_mode.bc/2019-11-12_21.33.22.040142/chkpt/chkpt_149.pt'
        weights = 'logdir/dataset_folder.dataset_toy4_pomdp.True_graphsteps.3_training_mode.bc/2019-11-13_19.24.37.715547/chkpt/chkpt_149.pt'
    else:
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.bc/2019-11-12_21.33.57.388801/chkpt/chkpt_149.pt'
        # NO INVERTING EDGES
        weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.bc/2019-11-13_19.25.08.723550/chkpt/chkpt_149.pt'

        # INVERTING EDGES
        # weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.bc_invertedge.True/2019-11-19_14.55.41.122930/chkpt/chkpt_149.pt'

        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.pg/offp.False_eps.0.2_gamma.0.7/2019-11-14_00.45.32.080556/chkpt/chkpt_149.pt'
        #weights = 'logdir/dataset_folder.dataset_toy4_pomdp.False_graphsteps.3_training_mode.pg_invertedge.True/offp.False_eps.0.2_gamma.0.7/2019-11-19_16.00.11.419509/chkpt/chkpt_149.pt'

    first_epoch = 0
    if weights is not None:
        print('Loading weights')
        state_dict = torch.load(weights)
        if args.continueexec:
            optimizer.load_state_dict(state_dict['optim_params'])
            first_epoch = state_dict['epoch']
        policy_net.load_state_dict(state_dict['model_params'])

    metrics = utils.AvgMetrics(['LCS', 'ActionLCS', 'O1LCS', 'O2LCS'], ':.2f')
    other_metrics = utils.AvgMetrics(['reward', 'success'], ':.2f')
    metrics_loss = utils.AvgMetrics(['PGLoss'], ':.3f')
    parameters = utils.AvgMetrics(['epoch', 'eps'], ':.2f')
    eps_value = args.eps_greedy
    print('Starting from epoch {} to {}'.format(first_epoch, args.num_epochs))
    for epoch in range(first_epoch, args.num_epochs):
        metrics.reset()
        metrics_loss.reset()
        other_metrics.reset()
        parameters.reset()

        eps_value = max(0, eps_value - 0.01*(epoch+1))

        random.shuffle(shuffle_indices)
        batched_indices = [shuffle_indices[i*args.batch_size:min((i+1)*args.batch_size, len(shuffle_indices))] for i in range(num_iter)]
        for it_i, indices in enumerate(batched_indices):
            agents = []
            for it, index_problem in enumerate(indices):
                problem = dataset.problems_dataset[index_problem]
                path_init_env = problem['graph_file']
                goal_str = problem['goal']
                if args.envstop and len(problem['program']) == 1:
                    # means that the program stops here, nothing to do
                    continue
                curr_envs[it].reset(path_init_env, {0:'(facing living_room[1] living_room[1])'})
                curr_envs[it].to_pomdp()
                agents.append(SingleAgent(curr_envs[it], goal_str, 0, dataset, policy_net))
            if len(agents) == 0:
                continue
            for agent in agents:
                actions_off_policy = None
                if args.off_policy:
                    actions_off_policy = utils.parse_prog(problem['program'])
                instructions, logits, r, o1l = agent.rollout(agent.goal, args.pomdp, actions_off_policy, args.eps_greedy, args.envstop)
                #print(logits)
            # For now gamma = 0.9 --> REDUCE
            success = 1. if r[-1] > 0 else 0.
            log_prob = torch.cat([x[0]+x[1]+x[2] for x in logits])
            dr = []
            gamma = args.gamma
            for t in range(len(r)):
                pw = 1.
                Gt = 0.
                for ri in range(t, len(r)):
                    Gt = Gt + gamma**pw * r[ri]
                    pw += 1
                dr.append(Gt)
            gt_instr = [x.lower() for x in problem['program']]
            gt_parsed = utils.parse_prog(gt_instr)
            pred_parsed = utils.parse_prog(instructions)
            lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS_multiple([gt_parsed], [pred_parsed])
            reward_tensor = torch.tensor(dr)[:, None].float().cuda()
            #std = reward_tensor.std() if len(dr) > 1 else 1.
            #reward_tensor = (reward_tensor - reward_tensor.mean()) / (1e-9 + std)

            # print(reward_tensor)
            #pdb.set_trace()
            optimizer.zero_grad()
            pg_loss = (-log_prob*reward_tensor).sum()

            pg_loss.backward()

            optimizer.step()

            metrics_loss.update({'PGLoss': pg_loss.detach().cpu().numpy()})
            parameters.update({'epoch': epoch, 'eps': eps_value})
            metrics.update({
                            'LCS': lcs_triple,
                            'ActionLCS': lcs_action,
                            'O1LCS': lcs_o1,
                            'O2LCS': lcs_o2})
            other_metrics.update({'success': success,
                                   'reward': torch.tensor(r).float().mean()})

            if it_i % helper.args.print_freq == 0:
                # print(r)
                # print(dr)

                print(problem['goal'])
                print(utils.pretty_print_program(pred_parsed, other=gt_parsed))
                # print(pred_parsed)
                print('Epoch:{}. Iter {}/{}.  Losses: {}\n'
                      'LCS: {}\nOther: {}'.format(
                    epoch, it_i, len(batched_indices),
                    str(metrics_loss),
                    str(metrics), str(other_metrics)))

                if not helper.args.debug:
                    helper.log(epoch*(len(batched_indices))+it_i, metrics, 'LCS', 'train', avg=False)
                    helper.log(epoch*(len(batched_indices))+it_i, other_metrics, 'other_metrics', 'train', avg=False)
                    helper.log(epoch*(len(batched_indices))+it_i, metrics_loss, 'Losses', 'train', avg=False)
                    helper.log(epoch * (len(batched_indices)) + it_i, parameters, 'parameters', 'train', avg=False)


        if (epoch + 1) % helper.args.save_freq == 0:
            weights_path = helper.save(epoch, 0., policy_net.state_dict(), optimizer.state_dict())
            test(helper.args, helper.dir_name, weights_path, epoch)


def test(args, path_name, weights, epoch):
    helper = utils.Helper(args, path_name)
    device = torch.device('cuda:0')
    batch_size = 1

    dataset_test = envdataset.EnvDataset(helper.args, 'test')
    policy_net = SinglePolicy(dataset_test)
    policy_net.cuda()

    policy_net = torch.nn.DataParallel(policy_net)
    if len(weights) > 0:
        state_dict = torch.load(weights)
        policy_net.load_state_dict(state_dict['model_params'])
    policy_net.eval()

    num_elems = len(dataset_test.problems_dataset)
    curr_envs = [gym.make('vh_graph-v0')]
    num_iter = num_elems

    shuffle_indices = list(range(num_elems))
    if path_name is not None:
        helper.log_text('test', 'Testing {}\n'.format(epoch))

    with torch.no_grad():
        metrics = utils.AvgMetrics(['LCS', 'ActionLCS', 'O1LCS', 'O2LCS'], ':.2f')
        other_metrics = utils.AvgMetrics(['reward', 'success'], ':.2f')
        metrics_loss = utils.AvgMetrics(['PGLoss'], ':.3f')

        batched_indices = [shuffle_indices[i * batch_size:min((i + 1) * batch_size, len(shuffle_indices))] for
                           i in range(num_iter)]

        for it_i, indices in enumerate(batched_indices):
            agents = []
            for it, index_problem in enumerate(indices):
                problem = dataset_test.problems_dataset[index_problem]
                path_init_env = problem['graph_file']
                goal_str = problem['goal']

                if args.envstop and len(problem['program']) == 1:
                    # means that the program stops here, nothing to do
                    continue
                curr_envs[it].reset(path_init_env, {0: '(facing living_room[1] living_room[1])'})
                curr_envs[it].to_pomdp()
                agents.append(SingleAgent(curr_envs[it], goal_str, 0, dataset_test, policy_net))

            if len(agents) == 0:
                continue

            for agent in agents:
                actions_off_policy = None
                if args.off_policy:
                    actions_off_policy = utils.parse_prog(problem['program'])
                instructions, logits, r, o1l = agent.rollout(agent.goal, args.pomdp, actions_off_policy, 0., args.envstop)


            success = 1. if r[-1] > 0 else 0.
            log_prob = torch.cat([x[0] + x[1] + x[2] for x in logits])
            dr = []
            gamma = args.gamma
            for t in range(len(r)):
                pw = 1.
                Gt = 0.
                for ri in range(t, len(r)):
                    Gt = Gt + gamma ** pw * r[ri]
                    pw += 1
                dr.append(Gt)

            gt_instr = [x.lower() for x in problem['program']]
            gt_parsed = utils.parse_prog(gt_instr)
            pred_parsed = utils.parse_prog(instructions)

            lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS_multiple([gt_parsed], [pred_parsed])
            reward_tensor = torch.tensor(dr)[:, None].float().cuda()
            # std = reward_tensor.std() if len(dr) > 1 else 1.
            # reward_tensor = (reward_tensor - reward_tensor.mean()) / (1e-9 + std)

            pg_loss = (-log_prob * reward_tensor).sum()

            metrics_loss.update({'PGLoss': pg_loss.detach().cpu().numpy()})
            metrics.update({
                            'LCS': lcs_triple,
                            'ActionLCS': lcs_action,
                            'O1LCS': lcs_o1,
                            'O2LCS': lcs_o2})
            other_metrics.update({'success': success,
                                  'reward': torch.tensor(r).float().mean()})

            if it_i % helper.args.print_freq == 0:
                if path_name is not None:
                    helper.log_text('test', 'Done epoch')
                    helper.log(epoch*(len(batched_indices))+it_i, metrics, 'LCS', 'test', avg=False)
                    helper.log(epoch*(len(batched_indices))+it_i, metrics_loss, 'Losses', 'test', avg=False)
                    helper.log(epoch*(len(batched_indices))+it_i, other_metrics, 'other_metrics', 'test', avg=False)
                    #helper.log(epoch * (len(batched_indices)) + it_i, parameters, 'parameters', 'test', avg=False)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    train()
    #curr_env = gym.make('vh_graph-v0')
    #dataset_agent()
    #interactive_agent()
