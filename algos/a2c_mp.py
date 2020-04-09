from utils.memory import MemoryMask
import torch
from torch import optim, nn
import time
import numpy as np
import pdb
import copy
import ray
import json
from utils.utils_models import Logger
from utils import utils_models
import models.actor_critic as actor_critic
from gym import spaces
import atexit

class A2C:
    def __init__(self, arenas, graph_helper, args):
        self.arenas = arenas
        base_kwargs = {
            'hidden_size': args.hidden_size,
            'max_nodes': args.max_num_objects,
            'num_classes': graph_helper.num_objects,
            'num_states': graph_helper.num_states

        }
        num_actions = graph_helper.num_actions
        action_space = spaces.Tuple((spaces.Discrete(num_actions), spaces.Discrete(args.max_num_objects)))

        self.device = torch.device('cuda:0' if args.cuda else 'cpu')

        # self.actor_critic = self.arenas[0].agents[0].actor_critic
        self.actor_critic = actor_critic.ActorCritic(action_space, base_name=args.base_net, base_kwargs=base_kwargs)

        self.actor_critic.to(self.device)

        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.args = args
        self.memory_all = []
        self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=args.lr)

        self.logger = None
        if args.logging:
            self.logger = Logger(args)

        atexit.register(self.close)

    def close(self):
        print('closing')
        del(self.arenas[0])
        pdb.set_trace()

    def rollout(self, logging=False, record=False):
        """ Reward for an episode, get the cum reward """

        # Reset hidden state of agents
        # TODO: uncomment
        async_routs = []
        for arena in self.arenas:
            async_routs.append(arena.rollout.remote(logging, record))

        info_envs = []
        for async_rout in async_routs:
            info_envs.append(ray.get(async_rout))

        # # TODO: comment
        # info_envs = []
        # info_envs.append(self.arenas[0].rollout(logging, record))

        rewards = []
        for process_data in info_envs:
            rewards.append(process_data[0])
            # successes.append(process_data[1]['success'])
            rollout_memory = process_data[2]

            # Add into memory
            for mem in rollout_memory[0]:
                self.memory_all.append(*mem)
            self.memory_all.append(None, None, None, 0, 0)

        # only get info form one process
        info_rollout = info_envs[0][1]
        return rewards, info_rollout


    def train(self, trainable_agents=None):

        self.memory_all = MemoryMask(self.memory_capacity_episodes)
        self.memory_all.reset()

        start_episode_id = 1
        start_time = time.time()
        total_num_steps = 0
        info_ep = []
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            eps = utils_models.get_epsilon(self.args.init_epsilon, self.args.final_epsilon, self.args.max_exp_episodes,
                                           episode_id)


            time_prerout = time.time()

            # TODO: Uncomment
            curr_model = self.actor_critic.state_dict()

            for k, v in curr_model.items():
               curr_model[k] = v.cpu()

            # ray.register_custom_serializer(torch.Tensor, serializer=serializer, deserializer=deserializer)
            # ray.register_custom_serializer(torch.LongTensor, serializer=serializer, deserializer=deserializer)
            # ray.register_custom_serializer(torch.FloatTensor, serializer=serializer, deserializer=deserializer)
            #
            m_id = ray.put(curr_model)

            # TODO: Uncomment
            ray.get([arena.set_weigths.remote(eps, m_id) for arena in self.arenas])
            c_r_all, info_rollout = self.rollout(logging=(episode_id % self.args.log_interval == 0))


            episode_rewards = c_r_all
            num_steps = info_rollout['nsteps']
            total_num_steps += num_steps

            end_time = time.time()

            action_space = info_rollout['action_space']
            obs_space = info_rollout['observation_space']
            successes = info_rollout['success']

            print("episode: #{} steps: {} reward: {} finished: {} FPS {} #Objects {} #Objects actions {}".format(
                episode_id, num_steps,
                [c_r_all[0][0]],
                info_rollout['success'],
                total_num_steps*1.0/(end_time-start_time), obs_space, action_space))

            if self.logger:
                if episode_id % self.args.log_interval == 0:

                    num_steps = info_rollout['nsteps']
                    epsilon = info_rollout['epsilon']
                    dist_entropy = (np.mean(info_rollout['entropy'][0]), np.mean(info_rollout['entropy'][1]))
                    # pdb.set_trace()
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

                    #list(self.env.task_goal[0].keys())
                    # self.env.env_id
                    # goal =
                    # apt = self.arena.get_goal

                    info_episodes = [{'success': successes,
                                      'goal': info_rollout['goals'][0],
                                      'apt': info_rollout['env_id']}]
                    self.logger.log_data(episode_id, total_num_steps, start_time, end_time, episode_rewards[0],
                                         dist_entropy, epsilon, successes, info_episodes)



            # ===================== off-policy training =====================
            if not self.args.on_policy and episode_id - start_episode_id + 1 >= self.args.replay_start:
                nb_replays = 1
                for replay_id in range(nb_replays):
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

                    hx = torch.zeros(N, self.actor_critic.hidden_size).to(self.device)

                    state_keys = trajs[0][0].state.keys()

                    for t in range(len(trajs) - 1):

                        # TODO: decompose here
                        inputs = {state_key: torch.cat([trajs[t][i].state[state_key] for i in range(N)]).to(self.device) for state_key in state_keys}

                        pdb.set_trace()
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
                        v, _, policy, hx = self.actor_critic.act(inputs, hx, mask, action_indices=action)


                        [array.append(element) for array, element in
                         zip((policies, actions, rewards, Vs, old_policies, dones, masks),
                             (policy, action, reward, v, old_policy, done, mask))]


                        dones.append(done)

                        if (t + 1) % self.args.t_max == 0:  # maximum bptt length
                            hx = hx.detach()


                    self._train(self.actor_critic,
                                self.optimizer,
                                policies,
                                Vs,
                                actions,
                                rewards,
                                dones,
                                masks,
                                old_policies,
                                verbose=1)

            if not self.args.debug and episode_id % self.args.save_interval == 0:
                self.logger.save_model(episode_id, self.actor_critic)


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
