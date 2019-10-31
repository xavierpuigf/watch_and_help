from models.single_policy import SinglePolicy

import torch
from torch import distributions
import pdb
import vh_graph
import gym
import envdataset
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

    def reset(self):
        self.agent_info = {
            'saved_log_probs': [],
            'indices': [],
            'action_space': [],
            'rewards': []
        }

    def single_step_probs(self, observation, belief, goal_name):
        """ Given an observation and a belief what are the chances of next step """





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


    def get_top_instruction(self, dataset, state, action_logits, o1_logits, o2_logits):
        pred_action = torch.argmax(action_logits, -1)
        pred_o1 = torch.argmax(o1_logits, -1)
        pred_o2 = torch.argmax(o2_logits, -1)
        object_ids = state[1]
        object_names = state[0]
        pred_instr = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                  [pred_action, pred_o1, pred_o2])
        print(pred_instr[0])

    def get_instruction(self, observations):
        indices = []
        space = []
        log_probs = []
        logit_o1, candidates_o1 = self.policy_net.get_first_obj(observations, self)
        # pdb.set_trace()
        distr_o1 = distributions.categorical.Categorical(logits=logit_o1)

        node_names = [x['class_name'] if type(x) == dict else x for x in candidates_o1]

        # pdb.set_trace()
        obj1_id = distr_o1.sample()
        object_1_selected = candidates_o1[obj1_id]
        space.append(candidates_o1)
        indices.append(obj1_id)
        log_probs.append(logit_o1)



        # If the object is None consider the none action, that means stop

        if candidates_o1[obj1_id]['class_name'] != 'stop':
            # Decide action
            action_candidates_tripl = self.env.get_action_space(obj1=object_1_selected, structured_actions=True)
            actions_unique = list(set([x[0] for x in action_candidates_tripl]))
            logits_action, candidates_action = self.policy_net.get_action(actions_unique, self, obj1_id)
            distr_a1 = distributions.categorical.Categorical(logits=logits_action)
            action_id = distr_a1.sample()
            space.append(candidates_action)
            indices.append(action_id)
            log_probs.append(logits_action)

            action_selected = actions_unique[action_id]
            action_candidates_tripl = self.env.get_action_space(action=action_selected,
                                                                obj1=object_1_selected,
                                                                structured_actions=True)
            if len(action_candidates_tripl) == 1:
                id_triple = 0 # There is no third object

            else:
                print(action_candidates_tripl)
                logits_triple, candidates_tripl = self.policy_net.get_second_obj(action_candidates_tripl, self)
                distr_triple = distributions.categorical(logits=logits_triple)
                id_triple = distr_triple.sample()
                space.append(candidates_tripl)
                indices.append(id_triple)
                log_probs.append(logits_triple)

            final_instruction = self.env.obtain_formatted_action(action_candidates_tripl[id_triple][0],
                                                                 action_candidates_tripl[id_triple][1:])
        else:
            final_instruction = '[stop]'

        # TODO: this should be done in a batch
        for it, lp in enumerate(log_probs):
            log_probs[it] = lp.log_softmax(0)
        dict_info = {
            'instruction': final_instruction,
            'log_probs': log_probs,
            'indices': indices,
            'action_space': space,

        }
        return dict_info

def interactive_agent():
    path_init_env = 'dataset_toy/init_envs/TrimmedTestScene3_graph_46.json'
    goal_name = '(facing living_room[1] living_room[1])'
    weights = 'logdir/pomdp.True_graphsteps.3/2019-10-30_17.35.51.435717/chkpt/chkpt_61.pt'

    curr_env.reset(path_init_env, goal_name)
    curr_env.to_pomdp()
    args = utils.read_args()
    args.max_steps = 1
    args.interactive = True
    helper = utils.Helper(args)
    dataset_interactive = envdataset.EnvDataset(args, process_progs=False)

    policy_net = SinglePolicy(dataset_interactive).cuda()
    policy_net = torch.nn.DataParallel(policy_net)
    state_dict = torch.load(weights)
    policy_net.load_state_dict(state_dict['model_params'])
    policy_net.eval()

    single_agent = SingleAgent(curr_env, goal_name, 0, policy_net)
    observations = single_agent.get_observations()
    gt_state = single_agent.env.vh_state.to_dict()
    nodes, edges, ids_used = dataset_interactive.process_graph(gt_state)
    print('Objects...')
    print([(x['id'], x['class_name']) for x in gt_state['nodes']])
    id_str = '135' # input('Id of object to find...')
    id_obj = int(id_str)
    goal_str = 'findnode_{}'.format(id_obj)
    id_char = dataset_interactive.object_dict.get_id('character')
    pdb.set_trace()
    while True:
        observations = single_agent.get_observations()
        gt_state = single_agent.env.vh_state.to_dict()

        # Really overkill if we only care about visibility
        nodes_visible, edges_visible, _ = dataset_interactive.process_graph(observations, ids_used)
        nodes_all, edges_all, _ = dataset_interactive.process_graph(gt_state, ids_used)
        visible_mask = nodes_visible[2]

        if not args.pomdp:
            class_names, object_ids, mask_nodes, state_nodes = nodes_all
            edges, edge_types, mask_edges = edges_all
        else:
            class_names, object_ids, mask_nodes, state_nodes = nodes_visible
            edges, edge_types, mask_edges = edges_visible

        graph_data = dataset_interactive.join_timesteps(class_names, object_ids, [state_nodes],
                                                        [edges], [edge_types], [visible_mask], mask_nodes, [mask_edges])

        goal_info = dataset_interactive.prepare_goal(goal_str, ids_used, class_names)
        # To tensor
        graph_data = [torch.tensor(x).unsqueeze(0) for x in graph_data]
        goal_info = [torch.tensor(x).unsqueeze(0) for x in list(goal_info)]
        pdb.set_trace()
        output = policy_net(graph_data, goal_info, id_char)
        action_logits, o1_logits, o2_logits, _ = output
        instruction = single_agent.get_top_instruction(dataset_interactive, graph_data, action_logits, o1_logits, o2_logits)
        print(instruction)
        input('Waiting for input...')


if __name__ == '__main__':
    curr_env = gym.make('vh_graph-v0')
    interactive_agent()
