import random
import pdb
import copy
import multiprocessing
from utils.utils import CloudpickleWrapper

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))

            elif cmd == 'get_observations':
                ob = env.get_observations()
                action_space = env.get_action_space()
                remote.send((ob, action_space))
            elif cmd == 'reset':

                ob = None
                while ob is None:
                    ob = env.reset(task_id=data)
                remote.send((ob, env.python_graph, env.task_goal))

            elif cmd == 'close':
                remote.close()
                break
            # elif cmd == 'get_spaces':
            #     remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class Arena:
    def __init__(self, agent_types, environment_fns, agent_env_mapping=None):
        # Subprocess start method
        forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
        start_method = 'forkserver' if forkserver_available else 'spawn'

        n_envs = len(environment_fns)

        self.waiting = {id: False for id in range(n_envs)}
        self.closed = False
        self.n_envs = n_envs
        ctx = multiprocessing.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, environment_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()


        self.agents = []
        for agent_type in agent_types:
            self.agents.append(agent_type)
        self.num_agents = len(agent_types)

        # if type(environment) == list:
        #     self.env = environment
        # else:
        #     self.env = [environment]

        if agent_env_mapping:
            self.agent_env_mapping = {agent_id: 0 for agent_id in range(len(self.agents))}
        else:
            self.agent_env_mapping = agent_env_mapping


    def close(self):
        if self.closed:
            return
        for remote_id, remote in enumerate(self.remotes):
            if self.waiting[remote_id]:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def reset(self, task_id=None):
        for remote in self.remotes:
            remote.send(('reset', task_id))

        obs = [remote.recv() for remote in self.remotes]
        for it, agent in enumerate(self.agents):
            env_id = self.agent_env_mapping[it]
            if agent.agent_type == 'MCTS':
                agent.reset(obs[env_id][1], obs[env_id][2], seed=it)
            else:
                agent.reset(obs[env_id][1])

    def get_actions(self, obs, action_space=None, env_id=0):
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}

        env_agent = self.envs[env_id]
        for it, agent in enumerate(self.agents):
            if self.agent_env_mapping[it] != env_id:
                continue
            env_agent = self.envs[self.agent_env_mapping[it]]
            if agent.agent_type == 'MCTS':
                opponent_subgoal = None
                if agent.recursive:
                    opponent_subgoal = self.agents[1 - it].last_subgoal
                dict_actions[it], dict_info[it] = agent.get_action(obs[it], env_agent.task_goal[it] if it == 0 else self.task_goal[it], opponent_subgoal)
            elif agent.agent_type == 'RL':

                dict_actions[it], dict_info[it] = agent.get_action(obs[it], env_agent.goal_spec if it == 0 else self.task_goal[it], action_space_ids=action_space[it])
        return dict_actions, dict_info


    def get_observations(self, env_id=None):
        if env_id is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = [env_id]

        for c_env_id in env_ids:
            curr_remote = self.remotes[c_env_id]
            curr_remote.send(('get_observations', None))

        obs = [remote.recv() for remote in self.remotes[c_env_id]]
        obs_per_env = {eid: ob for eid, ob in zip(env_ids)}
        return obs_per_env

    def step(self, env_id):
        # Sync
        obs_per_env = self.get_observations(env_id)

        for env_id, observation_env in obs_per_env.items():
            obs, action_space = observation_env
            dict_actions, dict_info = self.get_actions(obs, action_space, env_id)



        return info_per_env

    def step_async(self, action_envs, env_id=None):

        if env_id is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = [env_id]

        for c_env_id in env_ids:
            if c_env_id in action_envs:
                curr_remote = self.remotes[c_env_id]
                remote.send(('step', action_envs[c_env_id]))

        action_space = env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space, env_id)

        return env.step(dict_actions), dict_actions, dict_info

    def step_wait(self, env_id=None):
        if env_id is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = [env_id]

        obs, action_space = [], []
        for c_env_id in env_ids:
            curr_remote = self.remotes[c_env_id]
            resp = [remote]


    def step(self):
        info_env = {}

        for it, env in enumerate(self.env):
            info_env[it] = self.step_env(it)

        return info_env

    def run(self, random_goal=False, pred_goal=None):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """

        assert(len(self.env) == 1)
        self.task_goal = copy.deepcopy(self.env[0].task_goal)
        if random_goal:
            for predicate in self.env[0].task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        saved_info = {'task_id': self.env[0].task_id,
                      'env_id': self.env[0].env_id,
                      'task_name': self.env[0].task_name,
                      'gt_goals': self.env[0].task_goal[0],
                      'goals': self.task_goal[0],
                      'action': {0: [], 1: []}, 
                      'plan': {0: [], 1: []},
                      'subgoal': {0: [], 1: []},
                      # 'init_pos': {0: None, 1: None},
                      'finished': None,
                      'init_unity_graph': self.env[0].init_unity_graph,
                      'obs': []}
        success = False
        while True:
            (obs, reward, done, infos), actions, agent_info = self.step()[0]
            success = infos['finished']
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)
            for agent_id, info in agent_info.items():
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(info['plan'][:3])
                if 'subgoal' in info:
                    saved_info['subgoal'][agent_id].append(info['subgoal'][:3])
                if 'obs' in info:
                    saved_info['obs'].append(info['obs'])
            if done:
                break
        saved_info['finished'] = success
        return success, self.env[0].steps, saved_info
