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

class A2C:
    def __init__(self, arenas, graph_helper, args):
        self.arenas = arenas
        arena = self.arenas[0]
        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.args = args
        self.memory_all = []

        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.optimizers = [optim.RMSprop(agent.actor_critic.parameters(), lr=args.lr)
                           for agent in arena.agents]

        self.logger = None
        if args.logging:
            self.logger = Logger(args)

    def reset(self):
        super(A2C, self).reset()

    def rollout(self, logging=False, record=False):
        """ Reward for an episode, get the cum reward """


        # # TODO: comment
        info_envs = self.arenas[0].rollout(logging, record)
        rewards = []
        rewards = []
        process_data = info_envs

        rewards.append(process_data[0])
        rollout_memory = process_data[2]

        # Add into memory
        for mem in rollout_memory[0]:
            self.memory_all.append(*mem)
        self.memory_all.append(None, None, None, 0, 0)

        # only get info form one process:
        info_rollout = info_envs[1]
        return rewards, info_rollout


    def train(self, trainable_agents=None):
        if trainable_agents is None:
            trainable_agents = [0]

        self.memory_all = MemoryMask(self.memory_capacity_episodes)
        self.memory_all.reset()

        start_episode_id = 1
        start_time = time.time()
        total_num_steps = 0
        info_ep = []
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            eps = utils_models.get_epsilon(self.args.init_epsilon, self.args.final_epsilon, self.args.max_exp_episodes,
                                           episode_id)

            for agent in self.arenas[0].agents:
                if agent.agent_type == 'RL':
                    agent.epsilon = eps

            time_prerout = time.time()
            if episode_id >= 20:
                self.args.use_gt_actions = False
            c_r_all, info_rollout = self.rollout(logging=(episode_id % self.args.log_interval == 0))
            episode_rewards = c_r_all

            num_steps = info_rollout['nsteps']

            total_num_steps += num_steps


            action_space = []
            obs_space = []
            successes = []
            rewards = [c_r_all_roll[0] for c_r_all_roll in c_r_all]
            for info_rollout_ep in info_rollout:
                action_space.append(info_rollout_ep['action_space'])
                obs_space.append(info_rollout_ep['observation_space'])
                successes.append(info_rollout_ep['success'])

            end_time = time.time()
            print("episode: #{} steps: {} reward: {} finished: {} FPS {} #Objects {} #Objects actions {}".format(
                episode_id, num_steps,
                np.mean(rewards),
                info_rollout['success'],
                total_num_steps*1.0/(end_time-start_time), obs_space, action_space))

            if episode_id % self.args.log_interval == 0:
                script_done = info_rollout['script']
                script_tried = info_rollout['action_tried']

                print("Target:")
                print(info_rollout['target'][1])
                for iti, (script_t, script_d) in enumerate(zip(script_tried, script_done)):
                    info_step = ''
                    for relation in ['CLOSE', 'INSIDE', 'ON']:
                        if relation == 'INSIDE':
                            if len([x for x in info_rollout['step_info'][iti][1] if x[2] == relation]) == 0:
                                pdb.set_trace()

                        info_step += '  {}:  {}'.format(relation, ' '.join(['{}.{}'.format(x[0], x[1]) for x in info_rollout['step_info'][iti][1] if x[2] == relation]))

                    if script_d is None:
                        script_d = ''

                    if info_rollout['step_info'][iti][0] is not None:
                        char_info = '{:07.3f} {:07.3f}'.format(info_rollout['step_info'][iti][0]['center'][0], info_rollout['step_info'][iti][0]['center'][2])
                        print('{: <36} --> {: <36} | char: {}  {}'.format(script_t, script_d, char_info, info_step))
                    else:
                        print('{: <36} --> {: <36} |  {}'.format(script_t, script_d, info_step))


                if self.logger:
                    num_steps = info_rollout['nsteps']
                    epsilon = info_rollout['epsilon']
                    dist_entropy = (np.mean(info_rollout['entropy'][0]), np.mean(info_rollout['entropy'][1]))
                    info_episode = {
                        'success': successes,
                        'reward': c_r_all[0],
                        'script': info_rollout['script'],
                        'target': info_rollout['target'],
                        'info_step': info_rollout['step_info'],
                    }
                    if episode_id % max(self.args.log_interval, 10):
                        info_ep.append(info_episode)
                        file_name_log = '{}/{}/log.json'.format(self.logger.save_dir, self.logger.experiment_name)
                    with open(file_name_log, 'w+') as f:
                        f.write(json.dumps(info_ep, indent=4))
                    info_episodes = [{'success': successes,
                                      'goal': info_rollout['goals'][0],
                                      'apt': info_rollout['env_id']}]
                    self.logger.log_data(episode_id, total_num_steps, start_time, end_time, episode_rewards,
                                         dist_entropy, epsilon, successes, info_episodes)



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
                            trajs = self.memory_all.sample_batch_balanced(
                                self.args.batch_size,
                                self.args.neg_ratio,
                                maxlen=self.args.max_episode_length,
                                cutoff_positive=5.0)
                        else:
                            trajs = self.memory_all.sample_batch(
                                self.args.batch_size,
                                maxlen=self.args.max_episode_length)

                        N = len(trajs[0])
                        policies, actions, rewards, Vs, old_policies, dones, masks = \
                            [], [], [], [], [], [], []

                        hx = torch.zeros(N, self.arenas[0].agents[agent_id].hidden_size).to(self.device)

                        state_keys = trajs[0][0].state.keys()
                        for t in range(len(trajs) - 1):

                            # TODO: decompose here
                            inputs = {state_key: torch.cat([trajs[t][i].state[state_key] for i in range(N)]) for state_key in state_keys}

                            # TODO: delete

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
                            v, _, policy, hx = self.arenas[0].agents[agent_id].actor_critic.act(inputs, hx, mask, action_indices=action)


                            [array.append(element) for array, element in
                             zip((policies, actions, rewards, Vs, old_policies, dones, masks),
                                 (policy, action, reward, v, old_policy, done, mask))]


                            dones.append(done)

                            if (t + 1) % self.args.t_max == 0:  # maximum bptt length
                                hx = hx.detach()

                        self._train(self.arenas[0].agents[agent_id].actor_critic,
                                    self.optimizers[agent_id],
                                    policies,
                                    Vs,
                                    actions,
                                    rewards,
                                    dones,
                                    masks,
                                    old_policies,
                                    verbose=1,
                                    use_ce_loss=False)

            t_fb = time.time() - t_pfb
            print('Time analysis: #Steps {}. Rollout {}. Steps {}. Reset {}. Forward/Backward {}'.format(num_steps, t_rollout, t_steps, t_reset, t_fb))
            if not self.args.debug and episode_id % self.args.save_interval == 0:
                self.logger.save_model(episode_id, self.arenas[0].agents[0].actor_critic)


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
               verbose=0,
               use_ce_loss=False):
        """training"""

        off_policy = old_policies is not None
        policy_loss, value_loss, entropy_loss = 0, 0, 0
        ce_loss = 0
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

            # TODO: delete
            ce_loss += -log_prob.sum()
        if not args.no_time_normalization:
            policy_loss /= episode_length
            value_loss /= episode_length
            entropy_loss /= episode_length

        if verbose:
            print("policy_loss:", policy_loss.data.cpu().numpy()[0])
            print("value_loss:", value_loss.data.cpu().numpy()[0])
            print("entropy_loss:", entropy_loss.data.cpu().numpy())
            if use_ce_loss:
                print("crossentropy_loss:", ce_loss.data.cpu().numpy())
        # updating net
        optimizer.zero_grad()
        if not use_ce_loss:
            loss = policy_loss + value_loss + entropy_loss * args.entropy_coef
        else:
            loss = ce_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm, 1)
        optimizer.step()
