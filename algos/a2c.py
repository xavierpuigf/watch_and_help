from .arena import Arena
from utils.memory import MemoryMask
import torch
import numpy as np
import pdb
import copy
from torch import optim, nn

class A2C(Arena):
    def __init__(self, agent_types, environment, args):
        super(A2C, self).__init__(agent_types, environment)
        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.args = args
        self.memory_all = []
        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.optimizers = [optim.RMSprop(agent.actor_critic.parameters(), lr=args.lr)
                           for agent in agent_types]

    def reset(self):
        super(A2C, self).reset()

    def rollout(self, record=False):
        """ Reward for an episode, get the cum reward """

        # Reset hidden state of agents
        self.reset()
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0

        while not done and nb_steps < self.args.max_episode_length:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step()

            action_dict = {}
            nb_steps += 1
            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += reward

                # action_dict[agent_index] = agent_info[agent_index]['action']

            if record:
                actions.append(agent_actions)

            if not self.args.on_policy:
                # append to memory
                for agent_id in range(self.num_agents):
                    if self.agents[agent_id].agent_type == 'RL':
                        state = agent_info[agent_id]['state_inputs']
                        policy = [log_prob.data for log_prob in agent_info[agent_id]['probs']]
                        action = agent_info[agent_id]['actions']
                        rewards = reward

                        self.memory_all[agent_id].append(state, policy, action, rewards, 1)

        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        # padding
        # TODO: is this correct? Padding that is valid?
        while nb_steps < self.args.max_episode_length:
            nb_steps += 1
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL':
                    state = agent_info[agent_id]['state_inputs']
                    if 'edges' in obs.keys():
                        pdb.set_trace()
                    policy = [log_prob.data for log_prob in agent_info[agent_id]['probs']]
                    action = agent_info[agent_id]['actions']
                    rewards = reward
                    self.memory_all[agent_id].append(state, policy, action, 0, 0)

        if not self.args.on_policy:
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL' and \
                        len(self.memory_all[agent_id].memory[self.memory_all[agent_id].position]) > 0:
                    self.memory_all[agent_id].append(None, None, None, 0, 0)

        return c_r_all, success_r_all


    def train(self, trainable_agents=None):
        if trainable_agents is None:
            trainable_agents = [it for it, agent in enumerate(self.agents) if agent.agent_type == 'RL']

        self.memory_all = [MemoryMask(self.memory_capacity_episodes) for _ in range(self.num_agents)]
        cumulative_rewards = {agent_id: [] for agent_id in range(self.num_agents)}
        for memory in self.memory_all:
            memory.reset()

        start_episode_id = 1
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            c_r_all, success_r_all = self.rollout()
            print("episode: #{} steps: {} reward: {} finished: {}".format(
                episode_id, self.env.steps,
                [c_r_all[agent_id] for agent_id in trainable_agents],
                [success_r_all[agent_id] for agent_id in trainable_agents]))


            for agent_id in range(self.num_agents):
                cumulative_rewards[agent_id].append(c_r_all[agent_id])


            # ===================== off-policy training =====================
            if not self.args.on_policy and episode_id - start_episode_id + 1 >= self.args.replay_start:
                nb_replays = 1
                for replay_id in range(nb_replays):
                    for agent_id in trainable_agents:
                        if self.args.balanced_sample:
                            trajs = self.memory_all[agent_id].sample_batch_balanced(
                                self.args.batch_size,
                                self.args.neg_ratio,
                                maxlen=self.args.max_episode_length)
                        else:
                            trajs = self.memory_all[agent_id].sample_batch(
                                self.args.batch_size,
                                maxlen=self.args.max_episode_length)
                        N = len(trajs[0])
                        policies, actions, rewards, Vs, old_policies, dones, masks = \
                            [], [], [], [], [], [], []

                        hx = torch.zeros(N, self.agents[agent_id].hidden_size).to(self.device)

                        state_keys = trajs[0][0].state.keys()
                        for t in range(len(trajs) - 1):

                            # TODO: decompose here
                            inputs = {state_key: torch.cat([trajs[t][i].state[state_key] for i in range(N)]) for state_key in state_keys}

                            action = [torch.cat([torch.LongTensor([trajs[t][i].action[action_index]]).unsqueeze(0).to(self.device)
                                                for i in range(N)]) for action_index in range(2)]


                            old_policy = [torch.cat([trajs[t][i].policy[policy_index].to(self.device)
                                                    for i in range(N)]) for policy_index in range(2)]
                            done = torch.cat([torch.Tensor([trajs[t + 1][i].action is None]).unsqueeze(1).unsqueeze(
                                0).to(self.device)
                                              for i in range(N)])
                            mask = torch.cat([torch.Tensor([trajs[t][i].mask]).unsqueeze(1).to(self.device)
                                              for i in range(N)])
                            reward = np.array([trajs[t][i].reward for i in range(N)]).reshape((N, 1))

                            # policy, v, (hx, cx) = self.agents[agent_id].act(inputs, hx, mask)
                            v, _, policy, hx = self.agents[agent_id].actor_critic.act(inputs, hx, mask, action_indices=action)


                            [array.append(element) for array, element in
                             zip((policies, actions, rewards, Vs, old_policies, dones, masks),
                                 (policy, action, reward, v, old_policy, done, mask))]


                            dones.append(done)

                            if (t + 1) % self.args.t_max == 0:  # maximum bptt length
                                hx = hx.detach()
                        self._train(self.agents[agent_id].actor_critic,
                                    self.optimizers[agent_id],
                                    policies,
                                    Vs,
                                    actions,
                                    rewards,
                                    dones,
                                    masks,
                                    old_policies,
                                    verbose=1)

    def _train(self,
               model,
               optimizer,
               policies,
               Vs,
               actions,
               rewards,
               dones,
               masks,
               old_policies,
               verbose=0):
        """training"""

        off_policy = old_policies is not None
        policy_loss, value_loss, entropy_loss = 0, 0, 0
        args = self.args

        # compute returns
        episode_length = len(rewards)
        # print("episode_length:", episode_length)
        N = args.batch_size
        Vret = torch.from_numpy(np.zeros((N, 1))).float()

        for i in reversed(range(episode_length)):
            # v_next = Vs[i + 1].data.cpu().numpy()[0][0] if i < episode_length - 1 else 0
            # if off_policy:
            #     Vret = rewards[i] + args.discount * v_next * (1 - dones[i].data[0][0])
            # else:
            #     Vret = rewards[i] + args.discount * v_next
            Vret = torch.from_numpy(rewards[i]).float() + args.gamma * Vret
            A = Vret.to(self.device) - Vs[i]

            log_prob_action = policies[i][0].gather(1, actions[i][0]).log()
            log_prob_object = policies[i][1].gather(1, actions[i][1]).log()
            log_prob = log_prob_action + log_prob_object


            #print(log_prob_action, log_prob_object)
            if off_policy:
                log_prob_action_old = old_policies[i][0].gather(1, actions[i][0]).log()
                log_prob_object_old = old_policies[i][1].gather(1, actions[i][1]).log()
                log_prob_old = log_prob_action_old + log_prob_object_old

                rho = torch.exp(log_prob - log_prob_old).clamp(max=10.0)
            else:
                rho = 1.0

            if verbose > 1:
                print("Vret:", Vret)
                print("reward:", rewards[i])
                print("A:", A.data.cpu().numpy()[0][0])
                print("rho:", rho)

            num_masks = float(masks[i].sum(0).data.cpu().numpy()[0])

            single_step_policy_loss = -(log_prob \
                                        * A.data \
                                        * rho.data * masks[i]).sum(0) \
                                      / max(1.0, num_masks)

            if verbose > 1:
                print("single step policy loss:",
                      single_step_policy_loss.cpu().data.numpy()[0])

            policy_loss += single_step_policy_loss
            value_loss += (A ** 2 / 2 * masks[i]).sum(0) / max(1.0, num_masks)

            # Entropy for object and action
            entropy_loss += ((policies[i][0]+1e-9).log() * policies[i][0]).sum(1).mean(0)
            entropy_loss += ((policies[i][1]+1e-9).log() * policies[i][1]).sum(1).mean(0)

        if not args.no_time_normalization:
            policy_loss /= episode_length
            value_loss /= episode_length
            entropy_loss /= episode_length

        if verbose:
            print("policy_loss:", policy_loss.data.cpu().numpy()[0])
            print("value_loss:", value_loss.data.cpu().numpy()[0])
            print("entropy_loss:", entropy_loss.data.cpu().numpy())

        # updating net
        optimizer.zero_grad()
        loss = policy_loss + value_loss + entropy_loss * args.entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm, 1)
        optimizer.step()