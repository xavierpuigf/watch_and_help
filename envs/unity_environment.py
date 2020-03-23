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
                 num_agents=2,
                 max_episode_length=200,
                 env_task_set=None,
                 observation_types=None,
                 agent_goals=None,
                 use_editor=False,
                 base_port=8080,
                 port_id=0,
                 recording=False,
                 output_folder=None,
                 file_name_prefix=None,
                 executable_args={},
                 seed=123):

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.steps = 0
        self.env_id = None
        self.max_ids = {}

        self.pythnon_graph = None
        self.env_task_set = env_task_set

        self.num_agents = num_agents
        self.max_episode_length = max_episode_length

        self.recording = recording
        self.base_port = base_port
        self.port_id = port_id
        self.output_folder = output_folder
        self.file_nem_prefix = file_name_prefix

        self.default_width = 128
        self.default_height = 128
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = 1  # 0 TOP, 1 FRONT, 2 LEFT..

        if observation_types is not None:
            self.observation_types = observation_types
        else:
            self.observation_types = ['partial' for _ in range(num_agents)]

        if agent_goals is not None:
            self.agent_goals = agent_goals
        else:
            self.agent_goals = ['full' for _ in range(num_agents)]
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
            self.port_number = self.base_port + port_id


            self.comm = comm_unity.UnityCommunication(port=str(self.port_number), **executable_args)


        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)
        self.reset()

    def reward(self):
        count = 0
        done = True
        satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec)
        for key, value in satisfied.items():
            value_pred = min(len(value), self.goal_spec[key])
            mult = 1.0
            if 'hold' in key:
                mult = 10.
            if 'close' in key:
                mult = 0.1
            count += (value_pred*mult)
            if unsatisfied[key] > 0:
                done = False
        return count, done, {}

    def step(self, action_dict):
        script_list = utils.convert_action(action_dict)

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
            # pdb.set_trace()
            script_list = utils.convert_action({0: action_dict[0], 1: None})
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

        self.changed_graph = True


        reward, done, info = self.reward()
        obs = self.get_observations()
        graph = self.get_graph()
        self.python_graph_reset(graph)
        self.steps += 1
        info['finished'] = done
        if self.steps == self.max_episode_length:
            done = True
        # if done:
        #     pdb.set_trace()
        return obs, reward, done, info

    def python_graph_reset(self, graph):
        new_graph = utils.inside_not_trans(graph)
        self.python_graph = new_graph
        self.env.reset(new_graph, self.task_goal)
        self.env.to_pomdp()

    def get_goal(self, task_spec, agent_goal):
        if agent_goal == 'full':
            return task_spec
        elif agent_goal == 'grab':
            object_grab = random.choice([x.split('_')[1] for x,y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']])
            return {'holds_'+object_grab+'_'+'1': 1, 'close_'+object_grab+'_'+'1': 1}

        else:
            raise NotImplementedError

    def reset(self, environment_graph=None, task_id=None):

        # Make sure that characters are out of graph, and ids are ok
        if task_id is None:
            env_task = random.choice(self.env_task_set)
        else:
            env_task = self.env_task_set[task_id]

        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']

        self.task_name = env_task['task_name']

        old_env_id = self.env_id
        self.env_id = env_task['env_id']

        seed = (self.seed + self.task_id * 101) % 10007
        random.seed(seed)
        np.random.seed(seed)

        # TODO: in the future we may want different goals
        self.goal_spec = self.get_goal(self.task_goal[0], self.agent_goals[0])

        # if old_env_id == self.env_id:
        #     self.comm.fast_reset()
        # else:
        self.comm.reset(self.env_id)

        s,g = self.comm.environment_graph()
        if self.env_id not in self.max_ids.keys():
            max_id = max([node['id'] for node in g['nodes']])
            self.max_ids[self.env_id] = max_id

        max_id = self.max_ids[self.env_id]
        print(max_id)
        if environment_graph is not None:
            # TODO: this should be modified to extend well
            updated_graph = utils.separate_new_ids_graph(environment_graph, max_id)
            success, m = self.comm.expand_scene(updated_graph)
        else:
            updated_graph = utils.separate_new_ids_graph(env_task['init_graph'], max_id)
            success, m = self.comm.expand_scene(updated_graph)

        if not success:
            print("Error expanding scene")
            pdb.set_trace()
        self.offset_cameras = self.comm.camera_count()[1]

        if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
            rooms = random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        else:
            rooms = list(self.init_rooms)

        for i in range(self.num_agents):
            if i in self.agent_info:
                self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
            else:
                self.comm.add_character()

        _, self.init_unity_graph = self.comm.environment_graph()


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

    def get_action_space(self):
        dict_action_space = {}
        for agent_id in range(self.num_agents):
            if self.observation_types[agent_id] not in ['mcts', 'full']:
                raise NotImplementedError
            else:
                obs_type = 'mcts'
            visible_graph = self.get_observation(agent_id, obs_type)
            dict_action_space[agent_id] = [node['id'] for node in visible_graph['nodes']]
        return dict_action_space

    def get_observation(self, agent_id, obs_type, info={}):
        if obs_type == 'mcts':
            return self.env.get_observations(char_index=agent_id)

        elif obs_type == 'full':
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
