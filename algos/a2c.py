from .arena import Arena
from utils.memory import MemoryMask
import pdb

class A2C(Arena):
    def __init__(self, agent_types, environment, args):
        super(A2C, self).__init__(agent_types, environment)
        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.args = args
        self.memory_all = []

    def reset(self):
        super(A2C, self).reset()

    def rollout(self, record=False):
        """ Reward for an episode, get the cum reward """

        # Reset hidden state of agents
        self.reset()
        c_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0

        while not done and nb_steps < self.args.max_episode_length:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step()
            action_dict = {}
            nb_steps += 1
            print(nb_steps)
            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += reward

                # action_dict[agent_index] = agent_info[agent_index]['action']

            if record:
                actions.append(action_actions)

            if not self.args.on_policy:
                # append to memory
                for agent_id in range(self.num_agents):
                    if self.agents[agent_id].agent_type == 'RL':
                        state = obs[agent_id]
                        policy = [log_prob.data for log_prob in agent_info[agent_id]['log_probs']]
                        action = agent_actions[agent_id]
                        rewards = reward
                        self.memory_all[agent_id].append(state, policy, action, rewards, 1)

        # padding
        while nb_steps < self.args.max_episode_length:
            nb_steps += 1
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL':
                    state = obs[agent_id]
                    policy = [log_prob.data for log_prob in agent_info[agent_id]['log_probs']]
                    action = agent_actions[agent_id]
                    rewards = reward
                    self.memory_all[agent_id].append(state, policy, action, rewards, 1)

        if not self.args.on_policy:
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL' and \
                        len(self.memory_all[agent_id].memory[self.memory_all[agent_id].position]) > 0:
                    self.memory_all[agent_id].append(None, None, None, 0, 0)

        pdb.set_trace()
        return c_r_all


    def train(self, trainable_agents=None):
        if trainable_agents is None:
            trainable_agents = [it for it, agent in enumerate(self.agents) if agent.agent_type == 'RL']

        self.memory_all = [MemoryMask(self.memory_capacity_episodes) for _ in range(self.num_agents)]
        cumulative_rewards = {agent_id: [] for agent_id in range(self.num_agents)}
        for memory in self.memory_all:
            memory.reset()

        start_episode_id = 1
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            c_r_all = self.rollout()
            pdb.set_trace()
            print("episode: #{} steps: {} reward: {}".format(episode_id, self.env.steps, [c_r_all[agent_id] for agent_id in trainable_agents]))


            for agent_id in range(self.num_agents):
                cumulative_rewards[agent_id].append(c_r_all[agent_id])


            # ===================== off-policy training =====================
            pdb.set_trace()
            if not self.args.on_policy and episode_id - start_episode_id + 1 >= self.args.replay_start:
                nb_replays = 1
                for replay_id in range(nb_replays):
                    for agent_id in trainable_agents:
                        if self.args.balanced_sample:
                            trajs = self.memory_all[agent_id].sample_batch_balanced(
                                self.args.batch_size,
                                self.args.neg_ratio,
                                maxlen=args.max_episode_length)
                        else:
                            trajs = self.memory_all[agent_id].sample_batch(
                                self.args.batch_size,
                                maxlen=args.max_episode_length)

                        N = len(trajs[0])
                        policies, actions, rewards, Vs, old_policies, dones, masks = \
                            [], [], [], [], [], [], []

                        pdb.set_trace()