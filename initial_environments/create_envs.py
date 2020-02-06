import copy
import os
import sys
import ipdb
import json
import random
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])
sys.path.append(home_path+'/vh_mdp')
sys.path.append(home_path+'/virtualhome')
sys.path.append(home_path+'/vh_multiagent_models')
from simulation.unity_simulator import comm_unity as comm_unity


class EnvironmentCreator:
    def __init__(self, info, constraints, seed=0):
        self.constraints = constraints
        self.max_objects_same_type = 5
        self.seed = seed

        self.prob_modify_prior = 0.7

        # Probs inside
        self.prob_inside = 0.5
        self.prob_ontop = 0.5

        # The id where we start adding objects
        self.current_added_id = 1000
        self.classes_inside =  ['toilet',
                                'bathroom_cabinet',
                                'kitchencabinets',
                                'bathroom_counter',
                                'kitchencounterdrawer',
                                'cabinet',
                                'fridge',
                                'oven',
                                'dishwasher',
                                'microwave']

        self.fixed_objects = ['clothesshirt', 'clothespants', 'hanger']

        # self.fixed_objects = ['clothesshirt', 'clothespants']

    def get_grabbable_ids(self, environment):
        return [node['id'] for node in environment['nodes'] if 'GRABBABLE' in node['properties']]

    def get_surface_ids(self, environment):
        return [node['id'] for node in environment['nodes'] if 'SURFACES' in node['properties']]


    def get_container_ids(self, environment):
        return [node['id'] for node in environment['nodes'] if node['class_name'] in self.classes_inside]


    def transform_environment(self, environment):
        # Transforms the environment to satsfy the constraints
        pass

    def filter(self, environment):
        # Whether the environment can achieve the action
        return True

    def create_objects(self, example_node, num_objects):
        new_nodes = []
        for _ in range(num_objects):
            node_copy = copy.deepcopy(example_node)
            node_copy['id'] = self.current_added_id
            new_nodes.append(node_copy)
            self.current_added_id += 1
        return new_nodes

    def build_relations(self, nodes_place, nodes_inside, nodes_ontop, room_ids, constraints=None):
        edges = []

        for node in nodes_place:
            # Inside something or not
            inside_draw = random.uniform(0,1)
            ontop_draw = random.uniform(0,1)

            if inside_draw < self.prob_inside:
                container_id = random.choice(nodes_inside)
                relation_type = 'INSIDE'
            else: # ontop_draw < self.prob_ontop:
                # On top of something or not
                container_id = random.choice(nodes_ontop)
                relation_type = 'ON'

            # else:
            #     container_id = random.choice(room_ids)
            #     relation_type = 'INSIDE'
            
            edges.append({'from_id': node['id'], 'to_id': container_id, 'relation_type': relation_type})
        return edges




class SetupTableCreator(EnvironmentCreator):
    def __init__(self, info, constraints=None, seed=0):
        self.num_people = info['num_people']

        if constraints is None:
            constraints = {}
            constraints['prob_modify'] = {'wineglass': 0.95, 'plate': 0.95, 'forks': 0.95}

        super().__init__(info, constraints, seed)
        
    def return_objects_type(self, environment):
        glasses = [node for node in environment['nodes'] if node['class_name'] == 'wineglass']
        plates = [node for node in environment['nodes'] if node['class_name'] == 'plate']
        forks = [node for node in environment['nodes'] if 'fork' in node['class_name']]
        return {'glasses': glasses, 'plates': plates, 'forks': forks}

    def transform_environment(self, environment):
        all_nodes = environment['nodes']
        all_edges = []
        
        id2node = {node['id']: node for node in environment['nodes']}
        grabbable_ids = self.get_grabbable_ids(environment)
        room_ids = [node['id'] for node in graph['nodes'] if node['category'] == 'Rooms']
        nodes_inside = self.get_container_ids(environment)
        nodes_ontop = self.get_surface_ids(environment)
        
        assert(len(nodes_ontop) > 0)

        modified_ids = []

        dict_objects = self.return_objects_type(environment)
        object_class = ['wineglass', 'plate', 'fork']
        objects_place = [dict_objects['glasses'], dict_objects['plates'], dict_objects['forks']]

        # Select how many objects of each type to place
        range_place = [(max(0, self.num_people - len(x)), max(0, self.max_objects_same_type - len(x))) for x in objects_place]
        
        # Which nodes we will modify
        for grabbable_id in grabbable_ids:
            class_object = id2node[grabbable_id]['class_name']

            if class_object not in self.constraints['prob_modify']:
                prob_modify = self.prob_modify_prior
            else:
                prob_modify = self.constraints['prob_modify'][class_object]

            if class_object not in self.fixed_objects and random.uniform(0,1) < prob_modify:
                modified_ids.append(grabbable_id)

        # Use previous edges for non modified objects
        all_edges = [edge for edge in environment['edges'] if edge['from_id'] not in modified_ids and edge['to_id'] not in modified_ids]


        # Modify edges of some objects
        all_edges += self.build_relations([id2node[idi] for idi in modified_ids], nodes_inside, nodes_ontop, room_ids)

        # Place new objects in the scene
        for object_type_id in range(len(objects_place)):
            minim, maxim = range_place[object_type_id]
            minim = max(0, minim)
            maxim = max(0, maxim)
            if maxim == 0:
                continue

            num_objects_to_place = random.randrange(minim, maxim)
            
            if len(objects_place[object_type_id]) > 0:
                example_node = objects_place[object_type_id][0]
            else:
                print('Missing object {}'.format(object_class[object_type_id]))
                ipdb.set_trace()
                raise Exception
            node_objects = self.create_objects(example_node, num_objects_to_place)
            all_nodes += node_objects
            all_edges += self.build_relations(node_objects, nodes_inside, nodes_ontop, room_ids)

        
        return {'edges': all_edges, 'nodes': all_nodes}

    def filter(self, environment):
        dict_objects = self.return_objects_type(environment)
        
        return len(glasses) == self.num_people and len(plates) == self.num_people and len(forks) == self.num_people

if __name__ == '__main__':
    env_creator = SetupTableCreator(info={'num_people': 3})
    env_id = 0
    path_file = 'data/init_envs/{}.json'.format(env_id)
    comm = comm_unity.UnityCommunication()
    comm.reset(env_id)
    if not os.path.isfile(path_file):
        
        success, graph = comm.environment_graph()

        with open(path_file, 'w+') as f:
            f.write(json.dumps(graph, indent=4))
    else:
        with open(path_file, 'r') as f:
            graph = json.load(f)

    new_graph = env_creator.transform_environment(graph)
    success, message = comm.expand_scene(new_graph)
    ipdb.set_trace()

