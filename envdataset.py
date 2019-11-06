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
    def __init__(self, args, split='train', process_progs=True):
        self.dataset_file = args.dataset_folder
        self.scenes_split = {'train': [1,2,3,4,5], 'test': [6,7], 'all': [1,2,3,4,5,6,7]}
        self.split = split
        self.args = args
        self.process_progs = process_progs

        if self.process_progs:
            self.problems_dataset = self.read_problem(self.dataset_file, split)
            if args.overfit:
                self.problems_dataset = self.problems_dataset[:1]
                print(self.problems_dataset)
            self.num_items = len(self.problems_dataset)
        else:
            self.num_items = 0

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
        self.objects_remove = ['wall', 'floor', 'ceiling', 'door', 'curtain']

        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'close', 'facing', 'on']
        self.objects = self.getobjects()

        self.action_dict = DictObjId(self.actions + ['stop'])
        self.object_dict = DictObjId(self.objects + ['stop', 'no_obj'])
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)


        self.max_nodes = args.max_nodes #300
        self.max_edges = args.max_edges #500
        self.max_steps = args.max_steps # 10


        self.max_nodes_g = 0
        self.max_edges_g = 0

        self.node_stop = ('stop', -2)
        self.node_none = ('no_obj', -1)




    def __len__(self):
        return self.num_items

    def read_problem(self, folder_problem, split='train'):
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

            #
            #
            # # If there are 2 rooms one after another, then we should delete both
            # if len(program) >= 2:
            #     if '[WALK]' in program[0] and '[WALK]' in program[1]:
            #         ids = [int(x.split()[-1][1:-1]) for x in program[:2]]
            #         if ids[1] == location_char:
            #             program = program[2:]

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
            goal_class = class_names[id_in_model]
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
        program_info = self.prepare_program(problem['program'], ids_used)
        class_names = state_info[0]
        goal_info = self.prepare_goal(self.problems_dataset[idx]['goal'], ids_used, class_names[0])
        return state_info, program_info, goal_info



    def getobjects(self):
        object_file_name = '{}/obj_names.json'.format(self.dataset_file)
        print('Getting objects from {}...'.format(object_file_name))

        if not os.path.isfile(object_file_name):
            object_names = []
            all_problems = self.read_problem(self.dataset_file, 'all')
            for prob in tqdm(all_problems):
                with open(prob['graph_file'], 'r') as f:
                    graph = json.load(f)
                object_names += [x['class_name'] for x in graph['init_graph']['nodes']]
            object_names = list(set(object_names))
            object_names = [x for x in object_names if x not in self.objects_remove]
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
                o1[it] = self.node_none
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


    def obtain_graph_list(self, problem, program, full_init_env):
        if 'graphs_file' not in problem.keys():
            # Run the graph to get the info of the episode
            init_graph = problem['graph_file']
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
                full_state = curr_env.vh_state.to_dict()

                graphs = [(state, full_state)]
                try:
                    for instr in program[:-1]:

                        _, state, infos = curr_env.step(instr)
                        full_state = curr_env.vh_state.to_dict()
                        graphs.append((state, full_state))

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
        return graphs

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
            mask_nodes: which nodes exist at each step
            mask_edges: [max_steps, max_edges] which edges are valid
            ids_used: dict with a mapping from object ids to model ids
        '''

        # Build a priori the node names and node ids, for both seen and unseeen objects
        program = problem['program']
        init_graph = problem['graph_file']
        # Load the full graph and get the ids of objects
        with open(init_graph, 'r') as f:
            full_init_env = json.load(f)
            full_init_env = full_init_env['init_graph']

        node_info, _, ids_used = self.process_graph(full_init_env)
        class_names, object_ids, _, mask_nodes, _ = node_info

        graphs = self.obtain_graph_list(problem, program, full_init_env)

        info = [[], []]
        # The last instruction is the stop
        for state, state_full in graphs:

            self.max_nodes_g = max(self.max_nodes_g, len(state_full['nodes']))
            self.max_edges_g = max(self.max_edges_g, len(state_full['edges']))

            visible_ids = [x['id'] for x in state['nodes']]
            if self.args.pomdp:
                nodes, edges, _ = self.process_graph(state, ids_used, visible_ids)
            else:
                nodes, edges, _ = self.process_graph(state_full, ids_used, visible_ids)
            info[0].append(nodes)
            info[1].append(edges)

        _, _, visible_mask, _, state_nodes = [list(x) for x in zip(*info[0])]
        edges, edge_types, mask_edges = [list(x) for x in zip(*info[1])]

        graph_data = self.join_timesteps(class_names, object_ids, state_nodes,
                                         edges, edge_types, visible_mask, mask_nodes, mask_edges)


        return graph_data, ids_used

    def join_timesteps(self, class_names, object_ids, state_nodes,
                       edges, edge_types, visible_mask, mask_nodes, mask_edges):

        # Hack - they should not count on time
        object_ids = [object_ids]*len(state_nodes)
        class_names = [class_names]*len(state_nodes)

        class_names = np.concatenate([np.expand_dims(x, 0) for x in class_names])
        object_ids = np.concatenate([np.expand_dims(x, 0) for x in object_ids])
        state_nodes = np.concatenate([np.expand_dims(x, 0) for x in state_nodes])
        edges = np.concatenate([np.expand_dims(x, 0) for x in edges])
        edge_types = np.concatenate([np.expand_dims(x, 0) for x in edge_types])
        visible_mask = np.concatenate([np.expand_dims(x, 0) for x in visible_mask])

        # Pad to max steps
        remain = self.max_steps - len(state_nodes)
        class_names = np.pad(class_names, ((0, remain), (0, 0)), 'constant').astype(np.int64)
        object_ids = np.pad(object_ids, ((0, remain), (0, 0)), 'constant').astype(np.int64)

        state_nodes = np.pad(state_nodes, ((0, remain), (0, 0), (0, 0)), 'constant').astype(np.int64)
        edges = np.pad(edges, ((0, remain), (0, 0), (0, 0)), 'constant').astype(np.int64)
        edge_types = np.pad(edge_types, ((0, remain), (0, 0)), 'constant').astype(np.int64)

        visible_mask = np.pad(visible_mask, ((0, remain), (0, 0)), 'constant').astype(np.float32)
        mask_edges = np.pad(mask_edges, ((0, remain), (0, 0)), 'constant').astype(np.float32)

        return class_names, object_ids, state_nodes, edges, edge_types, visible_mask, mask_nodes, mask_edges

    def process_graph(self, graph, ids_used=None, visible_ids=None):
        """
        Maps the json graph into a dense structure to be used by the model
        :param graph: dictionary with the state, it can be POMDP or FOMDP
        :param ids_used: IDS mapping JSON nodes to MODEL ids
        :param visible_ids: list of ids that are visible from the ones in the graph. Mark how we mask the logits
        :return:
            (class_names
             object_ids
             visible_mask: [max_nodes] with 1 if visible
             mask_nodes: [max_nodes] with 1 if exists
             state_nodes: [max_nodes, num_states]
            )

            (edges: [max_edges, 3]
             edge_types: [max_edges]
             mask_edge: [max_edges]
            )
            ids_used: updated dict
        """
        if ids_used is None:
            ids_used = {}

        char_id = [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
        class_names = np.zeros(self.max_nodes)
        object_ids = np.zeros(self.max_nodes)
        mask_nodes = np.zeros(self.max_nodes).astype(np.float32)
        ids_remove = []
        for node in graph['nodes']:
            if node['class_name'] in self.objects_remove:
                ids_remove.append(node['id'])
            if node['id'] not in ids_used.keys():
                ids_used[node['id']] = len(ids_used.keys())

            object_ids[ids_used[node['id']]] = node['id']
            class_names[ids_used[node['id']]] = self.object_dict.get_id(node['class_name'])
            mask_nodes[ids_used[node['id']]] = 1

        # Include the final nodes
        # Include the Stop and None nodes
        if self.node_none[1] not in ids_used.keys():
            ids_used[self.node_none[1]] = len(ids_used.keys())
        if self.node_stop[1] not in ids_used.keys():
            ids_used[self.node_stop[1]] = len(ids_used.keys())

        object_ids[ids_used[self.node_none[1]]] = self.node_none[1]
        object_ids[ids_used[self.node_stop[1]]] = self.node_stop[1]

        class_names[ids_used[self.node_none[1]]] = self.object_dict.get_id(self.node_none[0])
        class_names[ids_used[self.node_stop[1]]] = self.object_dict.get_id(self.node_stop[0])

        mask_nodes[ids_used[self.node_none[1]]] = 1
        mask_nodes[ids_used[self.node_stop[1]]] = 1

        if visible_ids is None:
            visible_mask = mask_nodes
        else:
            visible_mask = np.zeros(self.max_nodes).astype(np.float32)
            for id_visible in visible_ids:
                visible_mask[ids_used[id_visible]] = 1
            visible_mask[ids_used[self.node_none[1]]] = 1
            visible_mask[ids_used[self.node_stop[1]]] = 1



        # Fill out state
        num_states = len(self.state_dict)
        state_nodes = np.zeros((self.max_nodes, num_states))
        state_ids = [(ids_used[x['id']], [self.state_dict.get_id(state.lower()) for state in x['states']]) for x in graph['nodes'] if
                     x['id'] in ids_used.keys()]

        for it, (id_model, node_states) in enumerate(state_ids):
            if len(node_states) > 0:
                state_nodes[id_model, node_states] = 1


        # Fill out edges
        edges = np.zeros((self.max_edges, 2))
        mask_edges = np.zeros(self.max_edges)
        edge_types = np.zeros(self.max_edges)
        room_ids = [x['id'] for x in graph['nodes'] if x['category'] == 'Rooms']

        ids_and_rooms = [(edge['from_id'], edge['to_id']) for edge in graph['edges'] if
                         edge['relation_type'] == 'INSIDE' and edge['to_id'] in room_ids]

        # Obtects inside rooms are also close to rooms
        # for from_id, to_id in ids_and_rooms:
        #    graph['edges'].append({'from_id': from_id, 'to_id': to_id, 'relation_type': 'CLOSE'})
        #    graph['edges'].append({'from_id': to_id, 'to_id': from_id, 'relation_type': 'CLOSE'})

        cont = 0
        for it, edge in enumerate(graph['edges']):
            if edge['from_id'] in ids_remove or edge['to_id'] in ids_remove:
                continue

            if edge['relation_type'] == 'CLOSE' and edge['from_id'] != char_id and edge['to_id'] != char_id:
                continue

            rel_id = self.relation_dict.get_id(edge['relation_type'].lower())

            id_from = ids_used[edge['from_id']]
            id_to = ids_used[edge['to_id']]
            edges[cont, :] = [id_from, id_to]
            edge_types[cont] = rel_id
            mask_edges[cont] = 1
            cont += 1

        return (class_names, object_ids, visible_mask, mask_nodes, state_nodes), \
               (edges, edge_types, mask_edges), ids_used


