from .base_environment import BaseEnvironment
from utils import utils_environment as utils

import sys
# sys.path.append('../../virtualhome/simulation/')
# sys.path.append('../../vh_mdp/')
from unity_simulator import comm_unity as comm_unity
from vh_graph.envs import belief, vh_env

import pdb
import random
import numpy as np

class UnityEnvironment(BaseEnvironment):

    def __init__(self,
                 env_id,
                 apartment_id,
                 num_agents,
                 env_task_set=None,
                 test_mode=False,
                 observation_types=None,
                 use_editor=False,
                 base_port=8080,
                 recording=False,
                 output_folder=None,
                 file_name_prefix=None,
                 executable_args={}):

        random.seed(env_id)
        np.random.seed(env_id)

        self.steps = 0

        self.pythnon_graph = None
        self.test_mode = test_mode
        self.env_task_set = env_task_set

        self.env_id = env_id
        self.apartment_id = apartment_id
        self.num_agents = num_agents

        self.recording = recording
        self.base_port = base_port
        self.output_folder = output_folder
        self.file_nem_prefix = file_name_prefix

        self.default_width = 128
        self.default_height = 128
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = 1  # 0 TOP, 1 FRONT, 2 LEFT..

        if observation_types is not None:
            self.observation_types = observation_types
        else:
            self.observation_types = ['partial', 'partial']

        self.agent_info = {
            0: 'Chars/Female1',
            1: 'Chars/Male1'
        }
        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {}

        self.changed_graph = False
        self.rooms = None
        self.id2node = None
        self.offset_cameras = None

        if use_editor:
            # Use Unity
            self.comm = comm_unity.UnityCommunication()
        else:
            # Launch the executable
            self.port_number = self.base_port + env_id


            self.comm = comm_unity.UnityCommunication(port=str(self.port_number), **executable_args)


        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)
        self.reset()

    def reward(self):
        count = 0
        done = True
        satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec)
        for key, value in satisfied.items():
            count += min(len(value), self.goal_spec[key])
            if unsatisfied[key] > 0:
                done = False
        return count, done, {}

    def step(self, action_dict):
        script_list = utils.convert_action(action_dict)

        reward, done, info = self.reward()
        if self.recording:
            success, message = self.comm.render_script(script_list,
                                                       recording=True,
                                                       gen_vid=False,
                                                       camera_mode='PERSON_TOP',
                                                       output_folder=self.output_folder,
                                                       file_name_prefix=self.file_name_prefix,
                                                       image_synthesis=['normal', 'seg_inst', 'seg_class'])
        else:
            # try:
            success, message = self.comm.render_script(script_list,
                                                       recording=False,
                                                       gen_vid=False,
                                                       processing_time_limit=20,
                                                       time_scale=10.)
        if not success:
            print(message)
            pdb.set_trace()

        self.changed_graph = True
        obs = self.get_observations()
        graph = self.get_graph()
        self.python_graph_reset(graph)
        self.steps += 1
        return obs, reward, done, info

    def python_graph_reset(self, graph):
        new_graph = utils.inside_not_trans(graph)
        self.python_graph = new_graph
        self.env.reset(new_graph, self.task_goal)
        self.env.to_pomdp()

    def reset(self, environment_graph=None):

        # Make sure that characters are out of graph, and ids are ok
        if self.test_mode:
            env_task = self.env_task_set[self.count_test]
        else:
            env_task = random.choice(self.env_task_set)
        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']

        # TODO: in the future we may want different goals
        self.goal_spec = self.task_goal[0]

        self.comm.reset(self.env_id)
        if environment_graph is not None:
            # TODO: this should be modified to extend well
            updated_graph = environment_graph
            self.expand_scene(updated_graph)

        self.offset_cameras = self.comm.camera_count()[1]
        for i in range(self.num_agents):
            if i in self.agent_info:
                self.comm.add_character(self.agent_info[i])
            else:
                self.comm.add_character()


        self.changed_graph = True
        graph = self.get_graph()
        self.python_graph_reset(graph)
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}

        obs = self.get_observations()
        self.steps = 0
        return obs

    def get_graph(self):
        if self.changed_graph:
            s, graph = self.comm.environment_graph()
            if not s:
                pdb.set_trace()
            self.graph = graph
            self.changed_graph = False
        return self.graph

    def get_observations(self):
        dict_observations = {}
        for agent_id in range(self.num_agents):
            obs_type = self.observation_types[agent_id]
            dict_observations[agent_id] = self.get_observation(agent_id, obs_type)
        return dict_observations


    def get_observation(self, agent_id, obs_type, info={}):
        if obs_type == 'partial':
            return self.env.get_observations(char_index=agent_id)

        elif obs_type == 'full':
            pdb.set_trace()
            return self.get_graph()

        elif obs_type == 'visible':
            raise NotImplementedError

        else:
            camera_ids = [self.offset_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
            if 'image_width' in info:
                image_width = info['image_width']
                image_height = info['image_height']
            else:
                image_width, image_height = self.default_width, self.default_height

            s, images = self.comm.camera_image(camera_ids, mode=obs_type, image_width=image_width, image_height=image_height)
            if not s:
                pdb.set_trace()
            return images[0]
