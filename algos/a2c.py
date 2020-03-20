from .arena import Arena
from utils.memory import MemoryMask
import torch
from torch import optim, nn
import time
import numpy as np
import pdb
import copy
import json
from utils.utils_models import Logger
from utils import utils_models

class A2C(Arena):
    def __init__(self, agent_types, environment, args):
        super(A2C, self).__init__(agent_types, environment)
        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.args = args
        self.memory_all = []
        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.optimizers = [optim.RMSprop(agent.actor_critic.parameters(), lr=args.lr)
                           for agent in agent_types]

        self.logger = None
        if args.logging:
            self.logger = Logger(args)

    def reset(self):
        super(A2C, self).reset()

    def rollout(self, logging=False, record=False):
        """ Reward for an episode, get the cum reward """

        # Reset hidden state of agents
        t1 = time.time()
        self.reset()
        t2 = time.time()
        t_reset = t2 - t1
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []

        info_rollout['step_info'] = []
        info_rollout['script'] = []

        # Target
        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec
            goal_class = list(pred.keys())[0].split('_')[1]
            id2node = {node['id']: node for node in init_graph['nodes']}
            info_goals = []
            info_goals.append([node for node in init_graph['nodes'] if node['class_name'] == goal_class])
            ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] == goal_class]
            info_goals.append([(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type'],
                                edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
            info_rollout['target'] = [pred, info_goals]

        while not done and nb_steps < self.args.max_episode_length:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step()
            if logging:
                node_id = [node['bounding_box'] for node in obs[0]['nodes'] if node['id'] == 1][0]
                edges_char = [(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type']) for edge in init_graph['edges'] if edge['from_id'] == 1]

                info_rollout['step_info'].append((node_id, edges_char))
                info_rollout['script'].append(agent_actions[0])

            nb_steps += 1
            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += reward
                # action_dict[agent_index] = agent_info[agent_index]['action']

            entropy_action.append(-((agent_info[0]['probs'][0]+1e-9).log()*agent_info[0]['probs'][0]).sum().item())
            entropy_object.append(-((agent_info[0]['probs'][1]+1e-9).log()*agent_info[0]['probs'][1]).sum().item())
            observation_space.append(agent_info[0]['num_objects'])
            action_space.append(agent_info[0]['num_objects_action'])
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
        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[0].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)
        info_rollout['t_reset'] = t_reset
        info_rollout['t_steps'] = t_steps

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

        return c_r_all, success_r_all, info_rollout


    def train(self, trainable_agents=None):
        if trainable_agents is None:
            trainable_agents = [it for it, agent in enumerate(self.agents) if agent.agent_type == 'RL']

        self.memory_all = [MemoryMask(self.memory_capacity_episodes) for _ in range(self.num_agents)]
        cumulative_rewards = {agent_id: [] for agent_id in range(self.num_agents)}
        for memory in self.memory_all:
            memory.reset()

        start_episode_id = 1
        start_time = time.time()
        total_num_steps = 0

        info_ep = []
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            eps = utils_models.get_epsilon(self.args.init_epsilon, self.args.final_epsilon, self.args.max_exp_episodes,
                                           episode_id)

            for agent in self.agents:
                if agent.agent_type == 'RL':
                    agent.epsilon = eps

            time_prerout = time.time()
            c_r_all, success_r_all, info_rollout = self.rollout(logging=(episode_id % 10 == 0))

            successes = info_rollout['success']
            num_steps = info_rollout['nsteps']
            epsilon = info_rollout['epsilon']
            obs_space = info_rollout['observation_space']
            action_space = info_rollout['action_space']
            dist_entropy = (np.mean(info_rollout['entropy'][0]), np.mean(info_rollout['entropy'][1]))

            episode_rewards = c_r_all
            total_num_steps += num_steps

            end_time = time.time()

            print("episode: #{} steps: {} reward: {} finished: {} FPS {} #Objects {} #Objects actions {}".format(
                episode_id, self.env.steps,
                [c_r_all[agent_id] for agent_id in trainable_agents],
                [success_r_all[agent_id] for agent_id in trainable_agents],
                total_num_steps*1.0/(end_time-start_time), obs_space, action_space))

            if self.logger:
                if episode_id % 10 == 0:
                    info_episode = {
                        'success': successes,
                        'reward': c_r_all[0],
                        'script': info_rollout['script'],
                        'target': info_rollout['target'],
                        'info_step': info_rollout['step_info'],
                    }
                    info_ep.append(info_episode)
                    file_name_log = '{}/{}/log.json'.format(self.logger.save_dir, self.logger.experiment_name)
                    with open(file_name_log, 'w+') as f:
                        f.write(json.dumps(info_ep, indent=4))
                info_episodes = [{'success': successes,
                                  'goal': list(self.env.task_goal[0].keys())[0],
                                  'apt': self.env.env_id}]
                self.logger.log_data(episode_id, total_num_steps, start_time, end_time, episode_rewards,
                                     dist_entropy, epsilon, successes, info_episodes)

            for agent_id in range(self.num_agents):
                cumulative_rewards[agent_id].append(c_r_all[agent_id])


            t_pfb = time.time()
            t_rollout = t_pfb - time_prerout
            t_steps = info_rollout['t_steps']
            t_reset = info_rollout['t_reset']
            # ===================== off-policy training =====================
            if not self.args.on_policy and episode_id - start_episode_id + 1 >= self.args.replay_start:
                nb_replays = 1
                for replay_id in range(nb_replays):
                    for agent_id in trainable_agents:
                        if self.args.balanced_sample:
                            trajs = self.memory_all[agent_id].sample_batch_balanced(
                                self.args.batch_size,
                                self.args.neg_ratio,
                                maxlen=self.args.max_episode_length,
                                cutoff_positive=5.0)
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

            t_fb = time.time() - t_pfb
            print('Time analysis: #Steps {}. Rollout {}. Steps {}. Reset {}. Forward/Backward {}'.format(self.env.steps, t_rollout, t_steps, t_reset, t_fb))
            if not self.args.debug and episode_id % self.args.save_interval == 0:
                self.logger.save_model(episode_id, self.agents[0].actor_critic)


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