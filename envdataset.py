from utils import DictObjId
import json
import pdb
from torch.utils.data import Dataset
import vh_graph
import gym
import utils
from tqdm import tqdm
import numpy as np
import os




class EnvDataset(Dataset):
    def __init__(self, args, split='train'):
        self.dataset_file = args.dataset_folder
        self.scenes_split = {'train': [1,2,3,4,5], 'test': [6,7]}
        self.split = split
        self.problems_dataset = self.read_problem(self.dataset_file, split)
        self.problems_dataset = self.problems_dataset
        print(self.problems_dataset[0]['goal'])
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

        self.action_dict = DictObjId(self.actions + ['stop'])
        self.object_dict = DictObjId(self.objects + ['stop', 'no_obj'])
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.num_items = len(self.problems_dataset)

        self.max_nodes = args.max_nodes #300
        self.max_edges = args.max_edges #500
        self.max_steps = args.max_steps # 10

        self.node_stop = ('stop', -2)
        self.node_none = ('no_obj', -1)

    def __len__(self):
        return self.num_items

    def read_problem(self, folder_problem, split='train'):
        # This should go in a dataset class
        file_problem = '{}/info.json'.format(folder_problem)
        with open(file_problem, 'r') as f:
            problems = json.load(f)

        scene_ids = self.scenes_split[split]
        problems = [x for x in problems if int(x['env_path'].split('_')[0].split('Scene')[-1]) in scene_ids]

        problems_dataset = []
        for id_problem, problem in enumerate(problems):
            graph_file = '{}/init_envs/{}'.format(folder_problem, problem['env_path'])
            goal_name = problem['goal']
            if '.txt' not in problem['program']:
                problem['program'] = problem['program'] + '.txt'
            program_file = '{}/programs/{}'.format(folder_problem, problem['program'])

            if 'file_name' in problem.keys():
                goal_file = '{}/{}'.format(folder_problem, problem['file_name'])
                with open(goal_file, 'r') as f:
                    goal_str = f.read()
            else:
                goal_str = goal_name

            with open(program_file, 'r') as f:
                program = f.readlines()
                program = [x.strip() for x in program]

            # program = [program[0]]
            program.append('[stop]')

            if not goal_str.lower().startswith('findnode'):
                continue

            # TODO: this should go in the data gen step
            with open(graph_file, 'r') as f:
                graph = json.load(f)
            graph = graph['init_graph']
            id_char = [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
            location_char = \
            [x['to_id'] for x in graph['edges'] if x['from_id'] == id_char and x['relation_type'] == 'INSIDE'][0]
            if '({})'.format(location_char) in program[0]:
                program = program[1:]

            problems_dataset.append(
                {
                    'id': id_problem,
                    'goal': goal_str,
                    'graph_file': graph_file,
                    'goal_name': goal_name,
                    'program': program
                }
            )

        return problems_dataset

    def prepare_goal(self, goal_str, ids_used, class_names):
        '''

        :param goal_str: the goal specification
        :param ids_used: the mapping from object ids in graph and model ids
        :return:
        '''
        if goal_str.lower().startswith('findnode'):
            # subgoal
            node_id = int(goal_str.split('_')[-1])
            id_in_model = ids_used[node_id]
            goal_id = 0
            goal_class = class_names[0, id_in_model]
            # This could be in the future all the nodes in the graph with the given id
            goal_node = ids_used[node_id]  # no obj

        elif goal_str.lower().startswith('findclass'):
            # subgoal
            class_name = goal_str.split('_')[-1]
            goal_id = 1
            goal_class = self.object_dict.get_id(class_name)
            # This could be in the future all the nodes in the graph with the given id
            goal_node = ids_used[-1]  # no obj

        else:
            # goal, we will need to figure out hot to handle
            goal_id = 2
            goal_class = self.object_dict.get_id('no_obj')
            goal_node = ids_used[-1] # no obj


        return goal_id, goal_class, goal_node

    def __getitem__(self, idx):
        problem = self.problems_dataset[idx]
        state_info, ids_used = self.prepare_data(problem)
        # pdb.set_trace()
        program_info = self.prepare_program(problem['program'], ids_used)
        class_names = state_info[0]
        goal_info = self.prepare_goal(self.problems_dataset[idx]['goal'], ids_used, class_names)
        return state_info, program_info, goal_info



    def getobjects(self):
        print('Getting objects...')
        object_file_name = '{}/obj_names.json'.format(self.dataset_file)

        if not os.path.isfile(object_file_name):
            object_names = []
            for prob in tqdm(self.problems_dataset):
                with open(prob['graph_file'], 'r') as f:
                    graph = json.load(f)
                object_names += [x['class_name'] for x in graph['init_graph']['nodes']]
            object_names = list(set(object_names))
            with open(object_file_name, 'w+') as f:
                f.write(json.dumps(object_names))
        else:
            with open(object_file_name, 'r') as f:
                object_names = json.load(f)
        object_names += ['stop']
        return object_names

    def prepare_program(self, program, ids_used):
        actions, o1, o2 = utils.parse_prog(program)

        # If there is a stop action, modify the object to be the stop_node
        for it, action in enumerate(actions):
            if action == 'STOP':
                o1[it] = self.node_stop
                o2[it] = self.node_none
        action_ids = np.array([self.action_dict.get_id(action) for action in actions])
        ob1 = np.array([ids_used[ob[1]] if ob is not None else ids_used[self.node_none[1]] for ob in o1])
        ob2 = np.array([ids_used[ob[1]] if ob is not None else ids_used[self.node_none[1]] for ob in o2])
        a = np.zeros(self.max_steps).astype(np.int64)
        o1 = np.zeros(self.max_steps).astype(np.int64)
        o2 = np.zeros(self.max_steps).astype(np.int64)
        a[:action_ids.shape[0]] = action_ids
        o1[:action_ids.shape[0]] = ob1
        o2[:action_ids.shape[0]] = ob2
        mask_steps = np.zeros(self.max_steps)
        mask_steps[:action_ids.shape[0]] = 1
        return a, o1, o2, mask_steps

    def prepare_data(self, problem):
        # TODO: clas_names, object_ids can eliminate the temporal dimension
        '''
        Given a problem with an intial env, returns a set of tensor describing the environment
        :param problem: dictionary with 'program' having the GT prog and 'graph_file' with the inital graph
                        if graphs_file exists, it will use that as the sequence of graphs, otherwise, it will create it
                        with the env.
        :return:

            class_names: [max_steps, max_nodes]: tensor with the id corresponding to the class name of every object
            object_ids: [max_steps, max_nodes]: tensor with the id of every object
            state_nodes: [max_steps, max_nodes, num_states]: binary tensor [i,j,k] = 1 if object j in step i has state k
            edges: [max_steps, max_edges, 2]
            edge_types: [max_steps, max_edges, 1]
            visible_mask: [max_steps, max_nodes] which nodes are visible at each step
            mask_edges: [max_steps, max_edges] which edges are valid
            ids_used: dict with a mapping from object ids to model ids
        '''

        # Build a priori the node names and node ids, for both seen and unseeen objects
        program = problem['program']
        init_graph = problem['graph_file']
        ids_used = {}
        # Load the full graph and get the ids of objects
        with open(init_graph, 'r') as f:
            full_init_env = json.load(f)
            full_init_env = full_init_env['init_graph']

        class_namenode_ids, id_nodes = [], []

        for it, node in enumerate(full_init_env['nodes']):
            ids_used[node['id']] = len(id_nodes)
            id_nodes.append(node['id'])
            class_namenode_ids.append(self.object_dict.get_id(node['class_name']))

        # Include the final nodes
        # Include the Stop and None nodes
        ids_used[self.node_stop[1]] = len(id_nodes)
        ids_used[self.node_none[1]] = len(id_nodes)+1

        id_nodes += [self.node_stop[1], self.node_none[1]]
        class_namenode_ids += [self.object_dict.get_id(self.node_stop[0]),
                               self.object_dict.get_id(self.node_none[0])]
        # pdb.set_trace()
        num_nodes = len(class_namenode_ids)
        class_names = np.zeros(self.max_nodes)
        object_ids = np.zeros(self.max_nodes)

        # Populate node_names [max_nodes]
        class_names[:num_nodes] = class_namenode_ids
        object_ids[:num_nodes] = id_nodes
        if 'graphs_file' not in problem.keys():
            # Run the graph to get the info of the episode
            graphs_file_name = init_graph[:-5] + '_multiple_{}.json'.format(problem['id'])

            if not os.path.isfile(graphs_file_name):
                goal = problem['goal']
                curr_env = gym.make('vh_graph-v0')

                if goal[0] != '(':
                    fnode = full_init_env['nodes'][0]
                    nnode = '{}[{}]'.format(fnode['class_name'], fnode['id'])
                    goal_name = '(facing {0} {0})'.format(nnode) # some random goal for now
                else:
                    goal_name = goal
                curr_env.reset(init_graph, goal_name)
                curr_env.to_pomdp()
                state = curr_env.get_observations()


                graphs = [state]
                try:
                    for instr in program[:-1]:

                        _, states, infos = curr_env.step(instr)
                        graphs.append(states)

                    with open(graphs_file_name, 'w+') as f:
                        f.write(json.dumps(graphs, indent=4))
                    problem['graphs_file'] = graphs_file_name
                except:

                    print('Error')
                    raise Exception
            else:
                problem['graphs_file'] = graphs_file_name
                with open(problem['graphs_file'], 'r') as f:

                    graphs = json.load(f)
        else:
            # load graphs
            with open(problem['graphs_file'], 'r') as f:
                graphs = json.load(f)

        info = []
        # The last instruction is the stop
        for state in graphs:
            info.append(self.process_graph(state, ids_used))

        state_nodes, edges, edge_types, visible_mask, mask_edges = zip(*info)

        # Hack - they should not count on time
        object_ids = [object_ids]*len(state_nodes)
        class_names = [class_names]*len(state_nodes)

        object_ids = np.concatenate([np.expand_dims(x, 0) for x in object_ids])
        class_names = np.concatenate([np.expand_dims(x, 0) for x in class_names])
        state_nodes = np.concatenate([np.expand_dims(x, 0) for x in state_nodes])
        edges = np.concatenate([np.expand_dims(x, 0) for x in edges])
        edge_types = np.concatenate([np.expand_dims(x, 0) for x in edge_types])

        visible_mask = np.concatenate([np.expand_dims(x, 0) for x in visible_mask])
        mask_steps = np.ones(len(program)).astype(np.int64)

        # Pad to max steps
        remain = self.max_steps - len(program)
        class_names = np.pad(class_names, ((0, remain), (0,0)), 'constant').astype(np.int64)
        object_ids = np.pad(object_ids, ((0, remain), (0, 0)), 'constant').astype(np.int64)

        state_nodes = np.pad(state_nodes, ((0, remain), (0,0), (0,0)), 'constant').astype(np.int64)
        edges = np.pad(edges, ((0, remain), (0, 0), (0,0)), 'constant').astype(np.int64)
        edge_types = np.pad(edge_types, ((0, remain), (0, 0)), 'constant').astype(np.int64)

        visible_mask = np.pad(visible_mask, ((0, remain), (0, 0)), 'constant').astype(np.float32)
        mask_edges = np.pad(mask_edges, ((0, remain), (0, 0)), 'constant').astype(np.float32)
        return (class_names, object_ids, state_nodes, edges, edge_types, visible_mask, mask_edges), ids_used

    def process_graph(self, state, ids_used):
        '''
        :param states: dictionary with the state
        :param ids_used: ids of nodes already used. Dictionary to id in the model
        :return:
            state_nodes: [max_nodes, num_states]
            edges: [max_edges, 3]
            edge_types: [max_edges]
            visible_mask: [max_nodes] with 1 if visible
            mask_edge: [max_edges]
        '''

        # TODO: this is computed 2 times...
        id_nodes = [x['id'] for x in state['nodes'] if x['id'] in ids_used.keys()]
        id_nodes += [self.node_stop[1], self.node_none[1]]
        ids_in_model = [ids_used[id] for id in id_nodes]
        #pdb.set_trace()
        # Populate state one hot [max_nodes, num_states]: one_hot
        num_states = len(self.state_dict)
        state_nodes = np.zeros((self.max_nodes, num_states))
        num_states = len(self.state_dict)

        state_ids = [[self.state_dict.get_id(state.lower()) for state in x['states']] for x in state['nodes'] if x['id'] in ids_used.keys()]
        for it, node_states in enumerate(state_ids):
            if len(node_states) > 0:
                state_nodes[ids_in_model[it], node_states] = 1


        # Populate edges one hot [max_nodes, max_nodes, edge_types]

        #edges = np.zeros((self.max_nodes, self.max_nodes, len(self.relation_dict)))
        edges = np.zeros((self.max_edges, 2))
        mask_edges = np.zeros(self.max_edges)
        edge_types = np.zeros(self.max_edges)

        cont = 0
        room_ids = [x['id'] for x in state['nodes'] if x['category'] == 'Rooms']
        char_id = [x['id'] for x in state['nodes'] if x['class_name'] == 'character']
        for it, edge in enumerate(state['edges']):
            rel_id = self.relation_dict.get_id(edge['relation_type'].lower())

            if edge['from_id'] not in ids_used.keys() or edge['to_id'] not in ids_used.keys():
                continue
            if edge['from_id'] == char_id and edge['to_id'] in room_ids and edge['relation_type'] == 'CLOSE':
                continue

            id_from = ids_used[edge['from_id']]
            id_to = ids_used[edge['to_id']]
            edges[cont, :] = [id_from, id_to]
            edge_types[cont] = rel_id
            mask_edges[cont] = 1
            cont += 1

        # Create

        visible_mask = np.zeros((self.max_nodes))
        visible_mask[ids_in_model] = 1
        # pdb.set_trace()
        return state_nodes, edges, edge_types, visible_mask, mask_edges