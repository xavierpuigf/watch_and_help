from utils import DictObjId
import json
import pdb
from torch.utils.data import Dataset, DataLoader
import vh_graph
import gym
import utils
import numpy as np

def read_problem(file_problem):
    # This should go in a dataset class
    with open(file_problem, 'r') as f:
        problems = json.load(f)

    problems_dataset = []
    for problem in problems:
        goal_file = problem['file_name']
        graph_file = problem['env_path']
        goal_name = problem['goal']
        program_file = problem['program']

        with open(goal_file, 'r') as f:
            goal_str = f.read()

        with open(program_file, 'r') as f:
            program = f.readlines()
            program = [x.strip() for x in program]

        program = [program[0]]
        # program.append('[stop]')

        problems_dataset.append(
            {
                'goal': goal_str,
                'graph_file': graph_file,
                'goal_name': goal_name,
                'program': program
            }
        )
    return problems_dataset


class EnvDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.problems_dataset = read_problem(dataset_file)

        self.actions = [
            "Walk",  # Same as Run
            # "Find",
            "Sit",
            "StandUp",
            "Grab",
            "Open",
            "Close",
            "PutBack",
            "PutIn",
            "SwitchOn",
            "SwitchOff",
            # "Drink",
            "LookAt",
            "TurnTo",
            # "Wipe",
            # "Run",
            "PutOn",
            "PutOff",
            # "Greet",
            "Drop",  # Same as Release
            # "Read",
            "PointAt",
            "Touch",
            "Lie",
            "PutObjBack",
            "Pour",
            # "Type",
            # "Watch",
            "Push",
            "Pull",
            "Move",
            # "Rinse",
            # "Wash",
            # "Scrub",
            # "Squeeze",
            "PlugIn",
            "PlugOut",
            "Cut",
            # "Eat",
            "Sleep",
            "WakeUp",
            # "Release"
        ]
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.objects = self.getobjects()

        self.action_dict = DictObjId(self.actions)
        self.object_dict = DictObjId(self.objects)
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.num_items = len(self.problems_dataset)

        self.max_nodes = 300
        self.max_edges = 500
        self.max_steps = 10

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        problem = self.problems_dataset[idx]
        state_info = self.prepare_data(problem)
        program_info = self.prepare_program(problem['program'])
        return state_info, program_info



    def getobjects(self):
        print('Getting objects...')
        object_names = []
        for prob in self.problems_dataset:
            with open(prob['graph_file'], 'r') as f:
                graph = json.load(f)
            object_names += [x['class_name'] for x in graph['init_graph']['nodes']]
        object_names = list(set(object_names))
        object_names += ['stop']
        return object_names


    def prepare_program(self, program):
        actions, o1, o2 = utils.parse_prog(program)
        action_ids = np.array([self.action_dict.get_id(action) for action in actions])
        ob1 = np.array([ob[1] if ob is not None else self.max_nodes-1 for ob in o1])
        ob2 = np.array([ob[1] if ob is not None else self.max_nodes-1 for ob in o2])
        a = np.zeros(self.max_steps).astype(np.int64)
        o1 = np.zeros(self.max_steps).astype(np.int64)
        o2 = np.zeros(self.max_steps).astype(np.int64)
        a[action_ids.shape[0]] = action_ids
        o1[action_ids.shape[0]] = ob1
        o2[action_ids.shape[0]] = ob2
        mask_steps = np.zeros(self.max_steps)
        mask_steps[:action_ids.shape[0]] = 1
        return a, o1, o2, mask_steps

    def prepare_data(self, problem):
        '''
        Given a problem with an intial env, returns a set of tenso describing the problem
        :param problem:
        :return:
        '''
        if 'graphs' not in problem.keys():
            graph_file = problem['graph_file']
            goal = problem['goal']
            graphs = []
            curr_env = gym.make('vh_graph-v0')
            curr_env.reset(graph_file, goal)
            curr_env.to_pomdp()
            program = problem['program']
            ids_used = {}
            info = []
            for instr in program:
                r, states, infos = curr_env.step(instr)
                info.append(self.process_graph(states, ids_used))

        class_names, state_nodes, edges, edge_types, visible_mask, mask_edges = zip(*info)
        class_names = np.concatenate([np.expand_dims(x, 0) for x in class_names])
        state_nodes = np.concatenate([np.expand_dims(x, 0) for x in state_nodes])
        edges = np.concatenate([np.expand_dims(x, 0) for x in edges])
        edge_types = np.concatenate([np.expand_dims(x, 0) for x in edge_types])

        visible_mask = np.concatenate([np.expand_dims(x, 0) for x in visible_mask])
        mask_steps = np.ones(len(program)).astype(np.int64)

        # Pad to max steps
        remain = self.max_steps - len(program)
        class_names = np.pad(class_names, ((0, remain), (0,0)), 'constant').astype(np.int64)
        state_nodes = np.pad(state_nodes, ((0, remain), (0,0), (0,0)), 'constant').astype(np.int64)
        edges = np.pad(edges, ((0, remain), (0, 0), (0,0)), 'constant').astype(np.int64)
        edge_types = np.pad(edge_types, ((0, remain), (0, 0)), 'constant').astype(np.int64)

        visible_mask = np.pad(visible_mask, ((0, remain), (0, 0)), 'constant').astype(np.float32)
        mask_edges = np.pad(mask_edges, ((0, remain), (0, 0)), 'constant').astype(np.float32)

        return class_names, state_nodes, edges, edge_types, visible_mask, mask_edges

    def process_graph(self, state, ids_used):
        '''
        :param states: dictionary with the state
        :param ids_used: ids of nodes already used. Dictionary to id in the model
        :return:
            class_name_ids: [max_nodes] class_name
            state_nodes: [max_nodes, num_states]
            edges: [max_edges, 3]
            visible_mask: [max_nodes] with 1 if visible
            mask_edge: [max_edges]
        '''

        # Build tensor of class_ids

        id_nodes = [x['id'] for x in state['nodes']]
        class_nodes = [x['class_name'] for x in state['nodes']]
        class_node_ids = [self.action_dict.get_id(cname) for cname in class_nodes]
        for id in id_nodes:
            if id not in ids_used.keys():
                ids_used[id] = len(ids_used.keys())

        ids_in_model = [ids_used[id] for id in id_nodes]
        class_name_ids = np.zeros(self.max_nodes)

        # Populate node_names [max_nodes]
        class_name_ids[ids_in_model] = class_node_ids

        # Populate state one hot [max_nodes, num_states]: one_hot
        num_states = len(self.state_dict)
        state_nodes = np.zeros((self.max_nodes, num_states))
        num_states = len(self.state_dict)
        state_ids = [[self.state_dict.get_id(state.lower()) for state in x['states']] for x in state['nodes']]
        for it, node_states in enumerate(state_ids):
            if len(node_states) > 0:
                state_nodes[ids_in_model[it], node_states] = 1


        # Populate edges one hot [max_nodes, max_nodes, edge_types]

        #edges = np.zeros((self.max_nodes, self.max_nodes, len(self.relation_dict)))
        edges = np.zeros((self.max_edges, 2))
        mask_edges = np.zeros(self.max_edges)
        edge_types = np.zeros(self.max_edges)
        for it, edge in enumerate(state['edges']):
            rel_id = self.relation_dict.get_id(edge['relation_type'].lower())
            id_from = ids_used[edge['from_id']]
            id_to = ids_used[edge['to_id']]
            edges[it, :] = [id_from, id_to]
            edge_types[it] = rel_id
            mask_edges[it] = 1
        visible_mask = np.zeros((self.max_nodes))
        visible_mask[ids_in_model] = 1

        return class_name_ids, state_nodes, edges, edge_types, visible_mask, mask_edges