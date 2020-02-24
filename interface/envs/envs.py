import numpy as np
from pathlib import Path
import cv2
import networkx as nx
from PIL import ImageFont, ImageDraw, Image
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import time
import logging
import atexit
from sys import platform
import subprocess
import os
import glob

import pdb
import random
import torch
import torchvision
import vh_graph
from vh_graph.envs import belief, vh_env
from simulation.unity_simulator import comm_unity as comm_unity

from agents import MCTS_agent, PG_agent
from gym import spaces, envs
import ipdb
from profilehooks import profile

import utils_rl_agent
logger = logging.getLogger("mlagents_envs")

def check_progress(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    id2node = {node['id']: node for node in state['nodes']}
    for key, value in goal_spec.items():
        elements = key.split('_')
        unsatisfied[key] = value if elements[0] in ['on', 'inside'] else 0
        satisfied[key] = [None] * 2
        satisfied[key]
        satisfied[key] = []
        for edge in state['edges']:
            if elements[0] in ['on', 'inside']:
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'offOn':
                if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[1] == 'offInisde':
                if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
    return satisfied, unsatisfied

class UnityEnvWrapper:
    def __init__(self, 
                 env_id,
                 env_copy_id, 
                 init_graph=None, 
                 file_name='../../executables/exec_linux02.10.x86_64', 
                 base_port=8080, 
                 num_agents=1,
                 recording=False,
                 output_folder=None,
                 file_name_prefix=None):
        atexit.register(self.close)
        self.port_number = base_port + env_copy_id 
        print(self.port_number)
        self.proc = None
        self.timeout_wait = 60
        self.file_name = file_name


        #self.launch_env(file_name
        self.output_folder = output_folder
        self.file_name_prefix = file_name_prefix

        # TODO: get rid of this, should be notfiied somehow else

        self.comm = comm_unity.UnityCommunication(port=str(self.port_number))
        print('Checking connection')
        # self.comm.check_connection()

        self.num_agents = num_agents
        self.graph = None
        self.recording = recording
        self.follow = False
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = 1 # 0 TOP, 1 FRONT, 2 LEFT..
        

        self.comm.reset(env_id)
        if init_graph is not None:
            self.comm.expand_scene(init_graph)

        # Assumption, over initializing the env wrapper, we only use one enviroment id
        # TODO: make sure this is true
        self.offset_cameras = self.comm.camera_count()[1]
        characters = ['Chars/Female1', 'Chars/Male1']
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])

        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}
        #comm.render_script(['<char0> [walk] <kitchentable> (225)'], camera_mode=False, gen_vid=False)
        #comm.render_script(['<char1> [walk] <bathroom> (11)'], camera_mode=False, gen_vid=False)  
        if self.follow:
            if self.recording:
                comm.render_script(['<char0> [walk] <kitchentable> (225)'], 
                                   recording=self.recording, 
                                   gen_vid=False, 
                                   camera_mode='FIRST_PERSON',
                                   output_folder=output_folder,
                                   file_name_prefix=file_name_prefix,
                                   image_synthesis=['normal', 'seg_inst', 'seg_class'])
            else:
                comm.render_script(['<char0> [walk] <kitchentable> (225)'], camera_mode=False, gen_vid=False)

        self.get_graph()
        #self.test_prep()

    def reset(self, env_id, init_graph=None):
        self.comm.reset(env_id)
        if init_graph is not None:
            self.comm.expand_scene(init_graph)
        self.offset_cameras = self.comm.camera_count()[1]
        characters = ['Chars/Female1', 'Chars/Male1']
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])

        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}
   
    def returncode_to_signal_name(returncode: int):
        """
        Try to convert return codes into their corresponding signal name.
        E.g. returncode_to_signal_name(-2) -> "SIGINT"
        """
        try:
            # A negative value -N indicates that the child was terminated by signal N (POSIX only).
            s = signal.Signals(-returncode)  # pylint: disable=no-member
            return s.name
        except Exception:
            # Should generally be a ValueError, but catch everything just in case.
            return None

    def close(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc = None
        return
        if self.proc is not None:
            # Wait a bit for the process to shutdown, but kill it if it takes too long
            try:
                self.proc.wait(timeout=self.timeout_wait)
                signal_name = self.returncode_to_signal_name(self.proc.returncode)
                signal_name = f" ({signal_name})" if signal_name else ""
                return_info = f"Environment shut down with return code {self.proc.returncode}{signal_name}."
                logger.info(return_info)
            except subprocess.TimeoutExpired:
                logger.info("Environment timed out shutting down. Killing...")
            # Set to None so we don't try to close multiple times.
            self.proc = None

    def launch_env(self, file_name, args=''):
        # based on https://github.com/Unity-Technologies/ml-agents/blob/bf12f063043e5faf4b1df567b978bb18dcb3e716/ml-agents/mlagents/trainers/learn.py
        cwd = os.getcwd()
        file_name = (
            file_name.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
        true_filename = os.path.basename(os.path.normpath(file_name))
        print(file_name)
        logger.debug("The true file name is {}".format(true_filename))
        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform == "darwin":
            candidates = glob.glob(
                os.path.join(
                    cwd, file_name + ".app", "Contents", "MacOS", true_filename
                )
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) > 0:
                launch_string = candidates[0]

        if launch_string is None:
            self.close()
            raise Exception(
                "Couldn't launch the {0} environment. "
                "Provided filename does not match any environments.".format(
                    true_filename
                )
            )
        else:
            docker_training = False
            if not docker_training:
                subprocess_args = [launch_string]
                #subprocess_args += ["-batchmode"]
                #subprocess_args += ["-http-port="+str(self.port_number)]
                subprocess_args += args
                try:
                    self.proc = subprocess.Popen(
                            subprocess_args, 
                            start_new_session=True)
                    ret_val = self.proc.poll()
                except:
                    raise Exception('Error, environment was found but could not be launched')
            else:
                raise Exception("Docker training is still not implemented")

        pass

    def get_graph(self):
        if True:  # self.graph is None:
            _, self.graph = self.comm.environment_graph()


        return self.graph

    # TODO: put in some utils
    def world2im(self, camera_data, wcoords):
        wcoords = wcoords.transpose()
        if len(wcoords.shape) < 2:
            return None
        proj = np.array(camera_data['projection_matrix']).reshape((4,4)).transpose()
        w2cam = np.array(camera_data['world_to_camera_matrix']).reshape((4,4)).transpose()
        cw = np.concatenate([wcoords, np.ones((1, wcoords.shape[1]))], 0) # 4 x N
        pixelcoords = np.matmul(proj, np.matmul(w2cam, cw)) # 4 x N
        pixelcoords = pixelcoords/pixelcoords[-1, :]
        pixelcoords = (pixelcoords + 1)/2.
        pixelcoords[1,:] = 1. - pixelcoords[1, :]
        return pixelcoords[:2, :]

    def get_visible_objects(self):
        camera_ids = [[self.offset_cameras+i*self.num_camera_per_agent+self.CAMERA_NUM for i in range(self.num_agents)][1]]
        object_ids = [int(idi) for idi in self.comm.get_visible_objects(camera_ids)[1].keys()]
        _, cam_data = self.comm.camera_data(camera_ids)
        graph = self.get_graph()
        object_position = np.array(
                [node['bounding_box']['center'] for node in graph['nodes'] if node['id'] in object_ids])
        obj_pos = self.world2im(cam_data[0], object_position)
        return object_ids, obj_pos

    def get_observations(self, mode='normal', image_width=128, image_height=128):
        camera_ids = [[self.offset_cameras+i*self.num_camera_per_agent+self.CAMERA_NUM for i in range(self.num_agents)][1]]
        s, images = self.comm.camera_image(camera_ids, mode=mode, image_width=image_width, image_height=image_height)
        #images = [image[:,:,::-1] for image in images]
        return images

    def test_prep(self):
        node_id_new = 2007
        s, graph = self.comm.environment_graph()
        table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
        container_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['fridge', 'freezer']][0]
        drawer_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['kitchencabinets']][0]


        id2node = {node['id']: node for node in graph['nodes']}

        # plates = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == table_id and id2node[edge['from_id']]['class_name'] == 'plate']
        # graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in plates and edge['to_id'] not in plates]
        # edge_plates = [{'from_id': plate_id, 'to_id': drawer_id, 'relation_type': 'INSIDE'} for plate_id in plates] 
        # graph['edges'] += edge_plates
        #self.comm.render_script(['<char0> [walk] <livingroom> (319)'], image_synthesis=[]).set_trace()


        new_node = {'id': node_id_new, 'class_name': 'glass', 'states': [], 'properties': ['GRABBABLE']}
        new_edge = {'from_id': node_id_new, 'relation_type': 'INSIDE', 'to_id': container_id}
        graph['nodes'].append(new_node)
        graph['edges'].append(new_edge)
        success = self.comm.expand_scene(graph)
        print(success)

    def agent_ids(self):
        return sorted([x['id'] for x in self.graph['nodes'] if x['class_name'] == 'character'])

    def set_record(self, 
                   output_folder,
                   file_name_prefix):
        self.output_folder = output_folder
        self.file_name_prefix = file_name_prefix

    def execute(self, actions): # dictionary from agent to action
        # Get object to interact with

        # This solution only works for 2 agents, we can scale it for more agents later

        agent_do = list(actions.keys())
        if self.follow:
            actions[0] = '[walk] <character> (438)'
        if len(actions.keys()) > 1:
            if sum(['walk' in x for x in actions.values()]) == 0:
                #continue
                objects_interaction = [x.split('(')[1].split(')')[0] for x in actions.values()]
                if len(set(objects_interaction)) == 1:
                    agent_do = [1] # [random.choice([0,1])]

        script_list = ['']
        for agent_id in agent_do:
            script = actions[agent_id]
            current_script = ['<char{}> {}'.format(agent_id, script)]
            

            script_list = [x+ '|' +y if len(x) > 0 else y for x,y in zip (script_list, current_script)]

        #if self.follow:
        script_list = [x.replace('[walk]', '[walktowards]') for x in script_list]
        # script_all = script_list
        self.graph = None
        if self.recording:
            success, message = self.comm.render_script(script_list,
                                                       recording=True, 
                                                       gen_vid=False, 
                                                       camera_mode='PERSON_TOP',
                                                       output_folder=self.output_folder,
                                                       file_name_prefix=self.file_name_prefix,
                                                       image_synthesis=['normal', 'seg_inst', 'seg_class'])
        else:
            success, message = self.comm.render_script(script_list, recording=False, gen_vid=False)
        if not success:
            ipdb.set_trace()
        result = {}
        for agent_id in agent_do:
            result[agent_id] = (success, message) 

        return result

    def is_terminal(self, goal_spec):
        _, unsatisfied = check_progress(self.graph, goal_spec)
        for predicate, count in unsatisfied.items():
            if count > 0:
                return False
        return True




class UnityEnv:
    def __init__(self, 
                 num_agents=2, 
                 seed=0, 
                 env_id=0, 
                 env_copy_id=0,
                 init_graph=None,
                 observation_type='coords', 
                 max_episode_length=100,
                 enable_alice=True,
                 simulator_type='python',
                 env_task_set=[],
                 task_type='complex',
                 max_num_objects=150,
                 logging=False,
                 recording=False,
                 record_dir=None):

        self.enable_alice = enable_alice
        self.task_type = task_type
        self.env_name = 'virtualhome'
        self.num_agents = num_agents
        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)
        self.env_copy_id = env_copy_id
        self.max_episode_length = max_episode_length
        self.simulator_type = simulator_type
        self.init_graph = init_graph
        self.task_goal = None
        self.env_task_set = env_task_set
        self.logging = logging
        self.recording = recording
        self.record_dir = record_dir

        self.unity_simulator = None # UnityEnvWrapper(int(env_id), int(env_copy_id), num_agents=self.num_agents)
        self.agent_ids =  [1,2] # self.unity_simulator.agent_ids()
        self.agents = {}

        self.system_agent_id = self.agent_ids[0]
        self.last_actions = [None] * self.num_agents
        self.last_subgoals = [None] * self.num_agents
        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {}

        if self.num_agents>1:
            self.my_agent_id = self.agent_ids[1]

        self.add_system_agent()

        self.actions = {}
        self.actions['system_agent'] = []
        self.actions['my_agent'] = []
        self.image_width = 224
        self.image_height = 224
        self.graph_helper = utils_rl_agent.GraphHelper(max_num_objects=max_num_objects,
                                                       simulator_type=simulator_type)


        ## ------------------------------------------------------------------------------------        
        self.observation_type = observation_type # Image, Coords
        self.viewer = None
        self.num_objects = max_num_objects

        num_actions = len(self.graph_helper.action_dict)
        num_object_classes = len(self.graph_helper.object_dict)
        self.action_space = spaces.Tuple((spaces.Discrete(num_actions), spaces.Discrete(self.num_objects)))



        if self.simulator_type == 'unity':
            if self.observation_type == 'coords':
                # current_obs = [current_obs, node_names, node_states, edges, edge_types, mask_nodes, mask_edges,
                #           rel_coords, position_objects, mask]
                self.observation_space = spaces.Dict({
                    # Image
                    'image': spaces.Box(low=0, high=255., shape=(3, self.image_height, self.image_width)),
                    # Graph
                    #utils_rl_agent.GraphSpace(),

                    'class_objects': spaces.Box(low=0, high=self.graph_helper.num_classes, shape=(self.graph_helper.num_objects, )),
                    'states_objects': spaces.Box(low=0, high=1., shape=(self.graph_helper.num_objects, self.graph_helper.num_states)),
                    'edge_tuples': spaces.Box(low=0, high=self.graph_helper.num_objects, shape=(self.graph_helper.num_edges, 2)),
                    'edge_classes': spaces.Box(low=0, high=self.graph_helper.num_edge_types, shape=(self.graph_helper.num_edges, )),
                    'mask_object': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_objects, )),
                    'mask_edge': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_edges, )),

                    # Target object
                    'object_dist': spaces.Box(low=-100, high=100, shape=(2,)),
                    'object_coords': spaces.Box(low=0, high=max(self.image_height, self.image_width),
                               shape=(self.num_objects, 2)), # 2D coords of the objects
                    'mask_position_objects': spaces.Box(low=0, high=1, shape=(self.num_objects, )),

                    'affordance_matrix': spaces.Box(low=0, high=1, shape=(num_actions, num_object_classes))
                })

            else:
                self.observation_space = spaces.Box(low=0, high=255., shape=(3, self.image_height, self.image_width))
        else:
            self.observation_space = spaces.Dict({
                'class_objects': spaces.Box(low=0, high=self.graph_helper.num_classes, shape=(self.graph_helper.num_objects,)),
                'states_objects': spaces.Box(low=0, high=1., shape=(self.graph_helper.num_objects, self.graph_helper.num_states)),
                'edge_tuples': spaces.Box(low=0, high=self.graph_helper.num_objects, shape=(self.graph_helper.num_edges, 2)),
                'edge_classes': spaces.Box(low=0, high=self.graph_helper.num_edge_types, shape=(self.graph_helper.num_edges,)),
                'mask_object': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_objects,)),
                'mask_edge': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_edges,)),
                'affordance_matrix': spaces.Box(low=0, high=1, shape=(num_actions, num_object_classes))
            })


        self.reward_range = (-10, 50.)
        self.metadata = {'render.modes': ['human']}
        self.spec = envs.registration.EnvSpec('virtualhome-v0')

        
        self.history_observations = []
        self.len_hist = 4
        self.num_steps = 0
        self.prev_dist = None

        self.micro_id = -1
        self.last_action = ''

        # The observed nodes
        self.nodes_visible = None

        self.info = {'dist': 0, 'reward': 0}

    def seed(self, seed):
        pass

    def close(self):
        self.unity_simulator.close()

    def distance_reward(self):
        dist = self.get_distance()


        reward = self.prev_dist - dist - 0.02
        #print(self.prev_dist, dist, reward)
        self.prev_dist = dist
        is_done = dist < 1.0
        if is_done:
            reward += 50
        info = {'dist': dist, 'done': is_done, 'reward': reward}
        return reward, is_done, info


    def reward(self):
        '''
        goal format:
        {predicate: number}
        predicate format:
            on_objclass_id
            inside_objclass_id 
        '''

        # Low level policy reward
        if self.task_type == 'find':
            return self.distance_reward()

        if self.simulator_type == 'unity':
            satisfied, unsatisfied = check_progress(self.unity_simulator.get_graph(), self.goal_spec)

        else:
            satisfied, unsatisfied = check_progress(self.env.state, self.goal_spec)


        # print('reward satisfied:', satisfied)
        # print('reward unsatisfied:', unsatisfied)
        # print('reward goal spec:', self.goal_spec)
        count = 0
        done = True
        for key, value in satisfied.items():
            count += min(len(value), self.goal_spec[key])
            if unsatisfied[key] > 0:
                done = False
        return count, done, {}
    

    def get_distance(self, target_id=None, target_class='microwave', norm=None):
        if self.simulator_type == 'unity':
            gr = self.unity_simulator.get_graph()
            if target_id is None:
                char_node = [node['bounding_box']['center'] for node in gr['nodes'] if node['class_name'] == 'character' and node['id'] == self.my_agent_id][0]
                target_node_id = [node['id'] for node in gr['nodes'] if node['class_name'] == target_class][0]
                target_id = target_node_id

            target_node = [node['bounding_box']['center'] for node in gr['nodes'] if node['id'] == target_id][0]
            if norm == 'no':
                return np.array(char_node) - np.array(target_node)
            dist = (np.linalg.norm(np.array(char_node) - np.array(target_node), norm))
            #print([node['id'] for node in gr['nodes'] if node['class_name'] == 'microwave'])
            # print(dist, char_node, micro_node)
        else:
            gr = self.env.state
            if target_id is None:
                target_id = [node['id'] for node in gr['nodes'] if node['class_name'] == target_class][0]

            if len([edge for edge in gr['edges'] if edge['from_id'] == target_id and edge['to_id'] == self.my_agent_id]) > 0:
                dist = 0
            else:
                dist = 5.
        return dist

    def render(self, mode='human'):
        image_width = 500
        image_height = 500
        obs, img = self.get_observations(mode='normal', image_width=image_width, image_height=image_height, drawing=True)

        fig = plt.figure()
        graph_viz = img[1][0].to_networkx()
        nx.draw(graph_viz, with_labels=True, labels=img[1][1])
        plt.show()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        img_graph = cv2.resize(image_from_plot, (image_width, image_height))
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        im_pil = Image.fromarray(img[0])

        draw = ImageDraw.Draw(im_pil)
        # Choose a font
        font = ImageFont.truetype("Roboto-Regular.ttf", 30)

        # Draw the text
        draw.text((0, 0), "dist: {:.3f}".format(self.info['dist']), font=font)
        draw.text((0, 60), "reward {:.3f}".format(self.info['reward']), font=font)
        draw.text((0, 90), "last action: {}".format(self.last_action), font=font)

        img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        img = np.concatenate([img, img_graph], 1)


        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=500)
            self.viewer.imshow(img)
            return self.viewer.isopen
         

    def reset(self, graph=None, task_goal=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
        env_task = random.choice(self.env_task_set)
        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']
        self.goal_spec = self.task_goal[self.system_agent_id]
        self.level = env_task['level']
        random.seed(self.task_id)
        np.random.seed(self.task_id)

        self.graph_helper.get_action_affordance_map(self.task_goal, {node['id']: node for node in self.init_graph['nodes']})
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])

        if self.simulator_type == 'unity':
            record_dir = self.record_dir

            if self.recording:
                Path(record_dir).mkdir(parents=True, exist_ok=True)
                file_name_prefix = str(self.task_id) + '_' + self.task_name
            else:
                record_dir = 'Output'
                file_name_prefix = None
            if self.unity_simulator is None:
                self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id),
                                                       init_graph=self.init_graph,
                                                       num_agents=self.num_agents,
                                                       recording=self.recording,
                                                       output_folder=record_dir + '/',
                                                       file_name_prefix=file_name_prefix)
            else:
                self.unity_simulator.set_record(output_folder=record_dir + '/', file_name_prefix=file_name_prefix)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]	        if graph is None:
            if True:
                self.unity_simulator.comm.fast_reset(self.env_id)
            else:
                self.unity_simulator.reset(self.env_id, self.init_graph)

            #self.env.reset(self.init_graph, self.task_goal)
            curr_graph_system_agent = self.inside_not_trans(self.unity_simulator.get_graph())
            self.init_unity_graph = self.get_unity_graph()

        else:
            # room_ids = [node['id'] for node in self.init_graph['nodes'] if node['category'] == 'Rooms']
            # random.choices(room_ids, k=2)
            room_ids = list(self.init_rooms)
            # ipdb.set_trace()
            char_positions = [{'from_id': 1,  'to_id': room_ids[0], 'relation_type': 'INSIDE'},
                              {'from_id': 2,  'to_id': room_ids[1], 'relation_type': 'INSIDE'}]
            dummybbox = {'center': [0,0,0], 'size': [0,0,0]}
            char_nodes = [{'id': 1, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []},
                          {'id': 2, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []}]
            init_graph_chars = {
                'edges': self.init_graph['edges'] + char_positions,
                'nodes': self.init_graph['nodes'] + char_nodes,
            }
            init_graph_chars['edges'] = [edge for edge in init_graph_chars['edges'] if edge['relation_type'] != 'CLOSE']
            graph = self.inside_not_trans(init_graph_chars)
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # print('unity env graph:', [edge for edge in self.init_graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            obs_n = self.env.reset(graph, self.task_goal)
            self.env.to_pomdp()
            curr_graph_system_agent = graph
            self.init_unity_graph = None
            # print('unity env graph:', [edge for edge in self.env.state['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # ipdb.set_trace()

        obs = self.get_observations()[0]

        self.agents[self.system_agent_id].reset(curr_graph_system_agent,
                                                self.task_goal,
                                                seed=self.system_agent_id,
                                                simulator_type=self.simulator_type)
        self.prev_dist = self.get_distance()
        self.num_steps = 0
        # pdb.set_trace()
        return obs

    def reset_2agents_python(self):
        env_task = random.choice(self.env_task_set)
        self.init_graph = env_task['init_graph']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']
        self.goal_spec = self.task_goal[self.system_agent_id]
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])
        if self.unity_simulator is None:
            self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id), init_graph=self.init_graph, num_agents=self.num_agents)
        graph = self.inside_not_trans(self.unity_simulator.get_graph())
        obs_n = self.env.reset(graph, self.task_goal)

        pdb.set_trace()
        self.agents[self.system_agent_id].reset(graph, self.task_goal, seed=self.system_agent_id)
        self.prev_dist = self.get_distance()
        self.num_steps = 0
        return obs_n

    def reset_MCTS(self, graph=None, task_goal=None, task_id=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
        if task_id is None:
            env_task = random.choice(self.env_task_set)
        else:
            env_task = self.env_task_set[task_id]
        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']
        self.goal_spec = self.task_goal[self.system_agent_id]
        self.level = env_task['level']
        random.seed(self.task_id)
        np.random.seed(self.task_id)

        self.graph_helper.get_action_affordance_map(self.task_goal, {node['id']: node for node in self.init_graph['nodes']})
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])


        if self.simulator_type == 'unity':
            record_dir = self.record_dir
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            file_name_prefix = str(self.task_id) + '_' + self.task_name

            if self.unity_simulator is None:
                self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id),
                                                       init_graph=self.init_graph,
                                                       num_agents=self.num_agents,
                                                       recording=self.recording,
                                                       output_folder=record_dir + '/',
                                                       file_name_prefix=file_name_prefix)
            else:
                self.unity_simulator.set_record(output_folder=record_dir + '/', file_name_prefix=file_name_prefix)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]           if graph is None:
            self.unity_simulator.reset(self.env_id, self.init_graph)

            #self.env.reset(self.init_graph, self.task_goal)
            curr_graph_system_agent = self.inside_not_trans(self.unity_simulator.get_graph())
            self.init_unity_graph = self.get_unity_graph()

        else:
            # room_ids = [node['id'] for node in self.init_graph['nodes'] if node['category'] == 'Rooms']
            # random.choices(room_ids, k=2)
            room_ids = list(self.init_rooms)
            # ipdb.set_trace()
            char_positions = [{'from_id': 1,  'to_id': room_ids[0], 'relation_type': 'INSIDE'},
                              {'from_id': 2,  'to_id': room_ids[1], 'relation_type': 'INSIDE'}]
            dummybbox = {'center': [0,0,0], 'size': [0,0,0]}
            char_nodes = [{'id': 1, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []},
                          {'id': 2, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []}]
            init_graph_chars = {
                'edges': self.init_graph['edges'] + char_positions,
                'nodes': self.init_graph['nodes'] + char_nodes,
            }
            init_graph_chars['edges'] = [edge for edge in init_graph_chars['edges'] if edge['relation_type'] != 'CLOSE']
            graph = self.inside_not_trans(init_graph_chars)
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # print('unity env graph:', [edge for edge in self.init_graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            obs_n = self.env.reset(graph, self.task_goal)
            self.env.to_pomdp()
            curr_graph_system_agent = graph
            self.init_unity_graph = None
            # print('unity env graph:', [edge for edge in self.env.state['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # ipdb.set_trace()

        self.agents[self.system_agent_id].reset(curr_graph_system_agent, self.task_goal, seed=self.system_agent_id, simulator_type=self.simulator_type)
        obs = None
        self.num_steps = 0
        # pdb.set_trace()
        return obs

    def reset_alice(self, graph=None, task_goal=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
        if graph is None:
            self.unity_simulator.comm.fast_reset(self.env_id)
        # #self.unity_simulator.comm.add_character()
        # #self.unity_simulator.comm.render_script(['<char0> [walk] <kitchentable> (225)'], gen_vid=False, recording=True)
        
        if task_goal is not None:
            self.goal_spec = task_goal[self.system_agent_id]
            self.task_goal = task_goal
            self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        self.prev_dist = self.get_distance()
        # obs = self.get_observations()[0]
        obs = None
        self.num_steps = 0
        return obs


    def get_action_command(self, my_agent_action):
        if my_agent_action is None:
            return None

        if self.simulator_type == 'unity':
            current_graph = self.unity_simulator.get_graph()
        else:
            current_graph = self.env.state

        objects1 = self.nodes_visible


        action = self.graph_helper.action_dict.get_el(my_agent_action[0][0])
        try:
            (o1, o1_id) = objects1[my_agent_action[1][0]]
        except:
            pdb.set_trace()
        #action_str = actions[my_agent_action]
        if o1 == 'no_obj':
            o1 = None

        converted_action = utils_rl_agent.can_perform_action(action, o1, o1_id, self.my_agent_id, current_graph)

        return converted_action

    def step(self, my_agent_action):
        #actions = ['<char0> [walktowards] <microwave> ({})'.format(self.micro_id), '<char0> [turnleft]', '<char0> [turnright]']
        if self.simulator_type == 'unity':
            action_dict = {}
            # system agent action

            if self.enable_alice:
                graph = self.get_graph()
                # pdb.set_trace()
                if self.num_steps == 0:
                    graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
                self.env.reset(graph , self.task_goal)
                system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
                self.last_actions[0] = system_agent_action
                self.last_subgoals[0] = system_agent_info['subgoals'][0]
                if system_agent_action is not None:
                    action_dict[0] = system_agent_action

            # user agent action
            action_str = self.get_action_command(my_agent_action)
            if action_str is not None:
                action_dict[1] = action_str
            dict_results = self.unity_simulator.execute(action_dict)
            self.num_steps += 1
            obs, _ = self.get_observations()

            reward, done, info = self.reward()
            dict_results['finished'] = done
            reward = torch.Tensor([reward])
            if self.num_steps >= self.max_episode_length:
                done = True
            done = np.array([done])
            graph = self.unity_simulator.get_graph()
            self.env.reset(graph, self.task_goal)
        else:
            action_dict = {}
            if self.enable_alice:
                graph = self.env.state
                if self.num_steps == 0:
                    graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (
                                edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
                self.env.reset(graph, self.task_goal)
                system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal,
                                                                                      self.last_actions[0],
                                                                                      self.last_subgoals[0])
                self.last_actions[0] = system_agent_action
                self.last_subgoals[0] = system_agent_info['subgoals'][0]
                if system_agent_action is not None:
                    action_dict[0] = system_agent_action
            action_str = self.get_action_command(my_agent_action)

            if action_str is not None:
                # if 'walk' not in action_str:
                print(action_str)
                action_dict[1] = action_str

            _, obs_n, dict_results = self.env.step(action_dict)
            obs, _ = self.get_observations()
            self.num_steps += 1
            reward, done, info = self.reward()
            dict_results['finished'] = done
            reward = reward# - 0.01
            reward = torch.Tensor([reward])
            if self.num_steps >= self.max_episode_length:
                done = True
            done = np.array([done])

        self.last_action = action_str
        return obs, reward, done, dict_results

    def step_2agents_python(self, action_dict):
        _, obs_n, info_n = self.env.step(action_dict)
        self.num_steps += 1
        reward, done, info = self.reward()
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        return obs_n, reward, done, info_n

    def step_with_system_agent_oracle(self, my_agent_action):
        #actions = ['<char0> [walktowards] <microwave> ({})'.format(self.micro_id), '<char0> [turnleft]', '<char0> [turnright]']
        action_dict = {}
        # system agent action
        graph = self.get_graph()
        # pdb.set_trace()
        if self.num_steps == 0:
            graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
        self.env.reset(graph , self.task_goal)
        system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
        self.last_actions[0] = system_agent_action
        self.last_subgoals[0] = system_agent_info['subgoals'][0]
        if system_agent_action is not None:
            action_dict[0] = system_agent_action
        system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
        self.last_actions[0] = system_agent_action
        self.last_subgoals[0] = system_agent_info['subgoals'][0]
        if system_agent_action is not None:
            action_dict[0] = system_agent_action
        # user agent action
        action_str = my_agent_action
        if action_str is not None:
            print(action_str)
            action_dict[1] = action_str

        dict_results = self.unity_simulator.execute(action_dict)
        self.num_steps += 1
        obs = None
        reward, done, info = self.reward()
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        # infos = {}
        #if done:
        #    obs = self.reset()
        return obs, reward, done, dict_results

    def step_alice(self):
        if self.simulator_type == 'unity':
            graph = self.get_graph()
            self.env.reset(graph, self.task_goal)
        self.num_steps += 1
        # obs, _ = self.get_observations()
        obs = None
        infos = {}
        reward, done, infos = self.reward()
        infos = {'finished': done}
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        
        #if done:
        #    obs = self.reset()
        return obs, reward, done, infos

    def add_system_agent(self):
        ## Alice model
        self.agents[self.system_agent_id] = MCTS_agent(unity_env=self,
                               agent_id=self.system_agent_id,
                               char_index=0,
                               max_episode_length=5,
                               num_simulation=100,
                               max_rollout_steps=3,
                               c_init=0.1,
                               c_base=1000000,
                               num_samples=1,
                               num_processes=1,
                               logging=self.logging)

    def get_system_agent_action(self, task_goal, last_action, last_subgoal, opponent_subgoal=None):
        if last_subgoal is not None:
            elements = last_subgoal.split('_')
            print(elements)
            # print(self.agents[self.system_agent_id].belief.edge_belief) #[int(elements[1])]['INSIDE']
            ipdb.set_trace()
        self.agents[self.system_agent_id].sample_belief(self.env.get_observations(char_index=0))
        self.agents[self.system_agent_id].sim_env.reset(self.agents[self.system_agent_id].previous_belief_graph, task_goal)
        action, info = self.agents[self.system_agent_id].get_action(task_goal[0], last_action, last_subgoal, opponent_subgoal)
        # if action == '[walk] <cutleryknife> (1010)':
        #     ipdb.set_trace()

        if action is None:
            print("system agent action is None! DONE!")
            # pdb.set_trace()
        # else:
        #     print(action, info['plan'])

        return action, info

    def get_all_agent_id(self):
        return self.agent_ids

    def get_my_agent_id(self):
        if self.num_agents==1:
            error("you haven't set your agent")
        return self.my_agent_id

    def get_graph(self):

        if self.simulator_type == 'unity':
            graph = self.unity_simulator.get_graph()
        else:
            graph = self.env.state
        graph = self.inside_not_trans(graph)
        return graph

    def get_unity_graph(self):
        return self.unity_simulator.get_graph()

    def get_system_agent_observations(self, modality=['rgb_image']):
        observation = self.agents[self.system_agent_id].num_cameras = self.unity_simulator.camera_image(self.system_agent_id, modality)
        return observation

    def get_my_agent_observations(self, modality=['rgb_image']):
        observation = self.agents[self.system_agent_id].num_cameras = self.unity_simulator.camera_image(self.my_agent_id, modality)
        return observation

    def inside_not_trans(self, graph):
        id2node = {node['id']: node for node in graph['nodes']}
        parents = {}
        grabbed_objs = []
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE':

                if edge['from_id'] not in parents:
                    parents[edge['from_id']] = [edge['to_id']]
                else:
                    parents[edge['from_id']] += [edge['to_id']]
            elif edge['relation_type'].startswith('HOLDS'):
                grabbed_objs.append(edge['to_id'])

        edges = []
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
                if len(parents[edge['from_id']]) == 1:
                    edges.append(edge)
            else:
                edges.append(edge)
        graph['edges'] = edges
        parent_for_node = {}

        ## Check that each node has at most one parent
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
                    print('{} has > 1 parent'.format(edge['from_id']))
                    pdb.set_trace()
                    raise Exception
                parent_for_node[edge['from_id']] = edge['to_id']
        
        ## Check that all nodes except rooms have one parent
        nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
        nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
        nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
        if len(nodes_without_parent) > 0:
            for nd in nodes_without_parent:
                print(id2node[nd])
            pdb.set_trace()
            raise Exception
        return graph

    # def inside_not_trans(self, graph):
    #     inside_node = {}
    #     other_edges = []
    #     id2node = {node['id']: node for node in graph['nodes']}
    #     for edge in graph['edges']:
    #         if edge['relation_type'] == 'INSIDE':
    #             if edge['from_id'] not in inside_node:
    #                 inside_node[edge['from_id']] = []
    #             inside_node[edge['from_id']].append(edge['to_id'])
    #         else:
    #             other_edges.append(edge)
    #     # Make sure we make trasnsitive first
    #     inside_trans = {}
    #     def inside_recursive(curr_node_id):
    #         if curr_node_id in inside_trans:
    #             return inside_trans[node_id]
    #         if curr_node_id not in inside_node.keys():
    #             return []
    #         else:
    #             all_parents = []
    #             for node_id_parent in inside_node[curr_node_id]:
    #                 curr_parents = inside_recursive(node_id_parent)
    #                 all_parents += curr_parents

    #             if len(all_parents) > 0:
    #                 inside_trans[curr_node_id] = list(set(all_parents))
    #             return all_parents

    #     for node_id in inside_node.keys():
    #         if len(inside_node[node_id]) > 1:
    #             inside_recursive(node_id)
    #         else:
    #             other_edges.append({'from_id':node_id, 'relation_type': 'INSIDE', 'to_id': inside_node[node_id][0]})

    #     num_parents = {}
    #     for node in graph['nodes']:
    #         if node['id'] not in inside_trans.keys():
    #             num_parents[node['id']] = 0
    #         else:
    #             num_parents[node['id']] = len(inside_trans[node['id']])

    #     edges_inside = []
    #     for node_id, nodes_inside in inside_trans.items():
    #         # all_num_parents = [num_parents[id_n] for id_n in nodes_inside]
    #         if len(nodes_inside) < 2:
    #             node_select = nodes_inside[0]
    #         else:
    #             node_select = [id_n for id_n in nodes_inside if id2node[id_n]['category'] != 'Rooms'][0]
    #         # max_np = max(all_num_parents)
    #         # node_select = [node_inside[i] for i, np in enumerate(all_num_parents) if np == max_np][0]
    #         edges_inside.append({'from_id':node_id, 'relation_type': 'INSIDE', 'to_id': node_select})
       
    #     graph['edges'] = edges_inside + other_edges

    #     edges_inside_aug = []
    #     for node in graph['nodes']:
    #         connected_edges = [edge for edge in graph['edges'] if edge['from_id'] == node['id']]
    #         count = sum([1 for edge in connected_edges if edge['relation_type'] == 'INSIDE'])
    #         if count == 0: # no inside node at all
    #             for edge in connected_edges:
    #                 if edge['relation_type'] == 'ON':
    #                     surface_id = edge['to_id']
    #                     room_id = None
    #                     for tmp_edge in graph['edges']:
    #                         if tmp_edge['from_id'] == surface_id and id2node[tmp_edge['to_id']]['category'] == 'Rooms':
    #                             room_id = tmp_edge['to_id']
    #                             break
    #                     if room_id is not None:
    #                         edges_inside_aug.append({'from_id': node['id'], 'relation_type': 'INSIDE', 'to_id': room_id})
    #                         break

    #     graph['edges'] += edges_inside_aug
    #     parent_for_node = {}

    #     ## Check that each node has at most one parent
    #     for edge in graph['edges']:
    #         if edge['relation_type'] == 'INSIDE':
    #             if edge['from_id'] in parent_for_node:
    #                 print('{} has > 1 parent'.format(edge['from_id']))
    #                 raise Exception
    #             parent_for_node[edge['from_id']] = edge['to_id']
        
    #     ## Check that all nodes except rooms have one parent
    #     nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
    #     nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
    #     if len(nodes_without_parent) > 0:
    #         for nd in nodes_without_parent:
    #             print(id2node[nd])
    #         pdb.set_trace()
    #         raise Exception
    #     return graph
   
    def get_observations(self, mode='seg_class', image_width=None, image_height=None, drawing=False):
        if self.simulator_type == 'unity':
            if image_height is None:
                image_height = self.image_height
            if image_width is None:
                image_width = self.image_width
            images = self.unity_simulator.get_observations(mode=mode, image_width=image_width, image_height=image_height)
            current_obs_img = images[0]
            current_obs_img = torchvision.transforms.functional.to_tensor(current_obs_img)[None, :]
            graph = self.unity_simulator.get_graph()

            distance = self.get_distance(norm='no')
            rel_coords = torch.Tensor(list([distance[0], distance[2]]))[None, :]
            visible_objects, position_objects = self.unity_simulator.get_visible_objects()
            id2node = {node['id']: node for node in graph['nodes']}
            visible_objects = [object for object in visible_objects if self.graph_helper.object_dict.get_id(id2node[object]['class_name']) != 0]

            position_objects_tensor = np.zeros((self.num_objects, 2))
            mask = np.zeros((self.num_objects))

            if position_objects is not None:
                position_objects = position_objects.transpose()
                mask[:position_objects.shape[0]] = 1
                position_objects_tensor[:position_objects.shape[0], :] = position_objects

            position_objects = torch.Tensor(position_objects_tensor)[None, :]
            mask = torch.Tensor(mask)[None, :]
            graph_inputs, graph_viz = self.graph_helper.build_graph(graph, ids=visible_objects, character_id=self.my_agent_id, plot_graph=drawing)
            self.nodes_visible = graph_viz[-1]

            current_obs = {'image': current_obs_img}
            current_obs.update(graph_inputs)
            current_obs.update(
                {
                    'affordance_matrix': self.graph_helper.obj1_affordance,
                    'object_dist': rel_coords,
                    'object_coords': position_objects,
                    'mask_position_objects': mask
                }
            )
            return current_obs, (images[0], graph_viz)

        else:
            obs = self.env.get_observations(char_index=1)
            class2id = {node['class_name']: node['id'] for node in obs['nodes']}
            category2id = {node['category']: node['id'] for node in obs['nodes']}

            ## filter graph
            obj_ids = []
            # class_types = ['character', 'kitchentable','coffeetable', 'kitchencounter', 'kitchencabinets', 'cabinet', 'bathroomcabinet', 'bookshelf', 'toilet', 'microwave', 'dishwahser', 'oven']
            class_types = ['character', 'kitchentable']
            for predicate in self.goal_spec:
                elements = predicate.split('_')
                class_types += list(elements[1:])
            if self.level == 0: # single room level
                obj_ids = [node['id'] for node in obs['nodes'] if str(node['id']) in class_types or \
                                            node['class_name'] in class_types]
            else:
                obj_ids = [node['id'] for node in obs['nodes'] if str(node['id']) in class_types or \
                                            node['class_name'] in class_types or \
                                            node['category'] == 'Rooms']
            filtered_obs = {
                'nodes': [node for node in obs['nodes'] if node['id'] in obj_ids],
                'edges': [edge for edge in obs['edges'] if edge['from_id'] in obj_ids and edge['to_id'] in obj_ids] 
            }
            # print([(node['id'], node['class_name'])for node in filtered_obs['nodes']])
            # ipdb.set_trace()

            graph_inputs, graph_viz = self.graph_helper.build_graph(filtered_obs,
                                                                    character_id=self.my_agent_id, plot_graph=drawing)


            current_obs = graph_inputs
            current_obs['affordance_matrix'] = self.graph_helper.obj1_affordance
            self.nodes_visible = graph_viz[-1]

            return current_obs, (None, graph_viz)

    def print_action(self, system_agent_action, my_agent_action):
        self.actions['system_agent'].append(system_agent_action)
        self.actions['my_agent'].append(my_agent_action)

        system_agent_actions = self.actions['system_agent']
        my_agent_actions = self.actions['my_agent']
        num_steps = len(system_agent_actions)

        print('**************************************************************************')
        if self.num_agents>1:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s \t\t\t\t\t\t |"my_agent": %s' % (i+1, system_agent_actions[i].ljust(30), my_agent_actions[i]))
        else:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s' % (i+1, system_agent_actions[i]))

        print('**************************************************************************')
