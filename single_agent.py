from models.single_policy import SinglePolicy

import torch
from torch import distributions
import pdb
import vh_graph
import gym
import json
import envdataset
import utils_viz
import utils


class SingleAgent():
    def __init__(self, env, goal, agent_id, policy=None):
        self.env = env
        self.goal = goal
        self.agent_id = agent_id
        self.policy_net = policy

        #if policy is not None:
        #    self.activation_info = policy.activation_info()
        self.beliefs = None

        self.agent_info = {
            'saved_log_probs': [],
            'indices': [],
            'action_space': [],
            'rewards': []
        }

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

    def sample_instruction(self, dataset, state, action_logits, o1_logits, o2_logits, pick_max=False):
        instruction = None
        object_ids = state[1]
        object_names = state[0]



        distr_object1 = distributions.categorical.Categorical(logits=o1_logits)
        if pick_max:
            obj1_id_model = torch.argmax(o1_logits, -1)
        else:
            obj1_id_model = distr_object1.sample()
        prob_1d = distr_object1.log_prob(obj1_id_model)
        obj1_id = state[1][0,0,obj1_id_model].item()

        if obj1_id < 0:
            # TODO: sometimes this none could correspond to standing up, we will need to solve this
            instruction = 'stop'
            action_candidate_ids = [dataset.action_dict.get_id('stop')]

        else:
            object1_node = [x for x in self.env.vh_state.to_dict()['nodes'] if x['id'] == obj1_id][0]
            # Given this object, get the action candidates
            action_candidates_tripl = self.env.get_action_space(obj1=object1_node, structured_actions=True)

            action_candidate_ids = [dataset.action_dict.get_id(x[0]) for x in action_candidates_tripl]

        mask = torch.zeros(action_logits[0,0].shape)
        mask[action_candidate_ids] = 1.

        action_logits_masked = action_logits.cpu() * mask + (-1e9) * (1-mask)
        distr_action1 = distributions.categorical.Categorical(logits=action_logits_masked)
        if pick_max:
            action_id_model = action_logits_masked.argmax(-1)
        else:
            action_id_model = distr_action1.sample()
        prob_action = distr_action1.log_prob(action_id_model)

        # Given action and object, get the last candidates
        if obj1_id < 0:
            obj2_id_cands = torch.tensor([dataset.node_none[1]])
        else:
            action_selected = dataset.action_dict.get_el(action_id_model.item())
            triple_candidates = self.env.get_action_space(action=action_selected,
                                                          obj1=object1_node,
                                                          structured_actions=True)

            obj2_id_cands = torch.tensor([dataset.node_none[1] if len(x) < 3 else x[2]['id'] for x in triple_candidates])
            if len(triple_candidates) == 0:
                pdb.set_trace()
        mask_o2 = (state[1] == obj2_id_cands).float()
        o2_logits_masked = (o2_logits.cpu() * mask_o2) + (1-mask_o2)*(-1e9)
        distr_object2 = distributions.categorical.Categorical(logits=o2_logits_masked)

        if pick_max:
            obj2_id_model = torch.argmax(o2_logits_masked, -1)
        else:
            obj2_id_model = distr_object2.sample()
        prob_2d = distr_object2.log_prob(obj2_id_model)
        #if instruction is None:
        instruction = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                   [action_id_model, obj1_id_model, obj2_id_model])
        instruction = instruction[0]
        return instruction, (prob_action, prob_1d, prob_2d)


    def obtain_logits_from_observations(self, dataset, curr_state, ids_used, visible_ids, class_names,
                                        object_ids, mask_nodes, goal_str):

        id_char = dataset.object_dict.get_id('character')
        nodes, edges, _ = dataset.process_graph(curr_state, ids_used, visible_ids)
        _, _, visible_mask, _, state_nodes = nodes
        edge_bin, edge_types, mask_edges = edges
        graph_data = dataset.join_timesteps(class_names, object_ids, [state_nodes],
                                                        [edge_bin], [edge_types], [visible_mask],
                                                        mask_nodes, [mask_edges])

        goal_info = dataset.prepare_goal(goal_str, ids_used, class_names)
        graph_data = [torch.tensor(x).unsqueeze(0) for x in graph_data]
        goal_info = [torch.tensor(x).unsqueeze(0) for x in list(goal_info)]

        output = self.policy_net(graph_data, goal_info, id_char)
        action_logits, o1_logits, o2_logits, _ = output
        mask_character = (graph_data[0] != id_char).float().cuda()
        o1_logits = o1_logits * mask_character + (1 - mask_character) * (-1e9)
        return graph_data, action_logits, o1_logits, o2_logits





def dataset_agent():
    args = utils.read_args()
    if args.pomdp:
        weights = 'logdir/dataset_folder.dataset_toy3_pomdp.True_graphsteps.3/2019-11-06_09.13.35.555005/chkpt/chkpt_49.pt'
    else:
        weights = 'logdir/dataset_folder.dataset_toy3_pomdp.False_graphsteps.3/2019-11-06_09.14.31.202175/chkpt/chkpt_49.pt'

    # 'logdir/pomdp.True_graphsteps.3/2019-10-30_17.35.51.435717/chkpt/chkpt_61.pt'

    # Set up the policy

    args.max_steps = 1
    args.interactive = True
    helper = utils.Helper(args)

    dataset = envdataset.EnvDataset(args, split='test')
    policy_net = SinglePolicy(dataset).cuda()
    policy_net = torch.nn.DataParallel(policy_net)
    policy_net.eval()

    if weights is not None:
        print('Loading weights')
        state_dict = torch.load(weights)
        policy_net.load_state_dict(state_dict['model_params'])

    final_list = []
    success, cont_episodes = 0, 0
    for problem in dataset.problems_dataset:
        path_init_env = problem['graph_file']
        goal_str = problem['goal']
        print('Goal: {}'.format(goal_str))
        goal_name = '(facing living_room[1] living_room[1])'
        curr_env.reset(path_init_env, {0: goal_name})
        curr_env.to_pomdp()
        single_agent = SingleAgent(curr_env, goal_name, 0, policy_net)


        gt_state = single_agent.env.vh_state.to_dict()
        node_id_char = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]
        # All the nodes
        nodes, _, ids_used = dataset.process_graph(gt_state)
        class_names, object_ids, _, mask_nodes, _ = nodes

        finished = False
        cont = 0
        while cont < 10 and not finished:
            if args.pomdp:
                curr_state = single_agent.get_observations()
                visible_ids = None
            else:
                curr_state = single_agent.env.vh_state.to_dict()
                visible_ids = single_agent.env.observable_object_ids_n[0]


            graph_data, action_logits, o1_logits, o2_logits = single_agent.obtain_logits_from_observations(
                dataset, curr_state, ids_used, visible_ids, class_names, object_ids, mask_nodes, goal_str)

            instruction, _ = single_agent.sample_instruction(dataset, graph_data, action_logits, o1_logits, o2_logits, pick_max=True)
            instr = list(zip(*instruction))[0]
            str_instruction = utils.pretty_instr(instr)
            #print(str_instruction)
            if str_instruction.strip() == '[stop]':
                finished = True
            else:
                single_agent.env.step({0: str_instruction})
            cont += 1

        goal_id = int(goal_str.split('_')[-1])
        edges_goal = [x for x in single_agent.env.vh_state.to_dict()['edges']
                      if x['relation_type'] == 'CLOSE' and x['from_id'] == goal_id and x['to_id'] == node_id_char]
        edge_found = len(edges_goal) > 0
        curr_success = False
        if goal_id in single_agent.env.observable_object_ids_n[0] and edge_found:
            success += 1
            curr_success = True

        final_list.append((path_init_env, goal_id, curr_success))
        cont_episodes += 1

        #print(success, cont_episodes)
    with open('output_{}.json'.format(args.pomdp), 'w+') as f:
        f.write(json.dumps(final_list))


# def policy_gradient():

def interactive_agent():
    path_init_env = 'dataset_toy3/init_envs/TrimmedTestScene6_graph_42.json'
    goal_name = '(facing living_room[1] living_room[1])'
    weights = 'logdir/dataset_folder.dataset_toy3_pomdp.False_graphsteps.3/2019-11-06_09.14.31.202175/chkpt/chkpt_31.pt'
    # 'logdir/pomdp.True_graphsteps.3/2019-10-30_17.35.51.435717/chkpt/chkpt_61.pt'

    # Set up the policy
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
    single_agent = SingleAgent(curr_env, goal_name, 0, policy_net)

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
        pdb.set_trace()
        if str_instruction.strip() == '[stop]':
            print('Episode finished')
        else:
            single_agent.env.step({0: str_instruction})
        pdb.set_trace()

if __name__ == '__main__':
    curr_env = gym.make('vh_graph-v0')
    dataset_agent()
    #interactive_agent()
