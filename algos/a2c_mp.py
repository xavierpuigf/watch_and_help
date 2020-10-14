from utils.memory import MemoryMask
import torch
from torch import optim, nn
import time
import numpy as np
import pdb
import copy
import ray
import json
import ipdb
from utils.utils_models import Logger
from utils import utils_models
import models.actor_critic as actor_critic
from models import actor_critic, actor_critic_hl_mcts
from gym import spaces
import atexit

class A2C:
    def __init__(self, arenas, graph_helper, args):
        self.arenas = arenas
        base_kwargs = {
            'hidden_size': args.hidden_size,
            'max_nodes': args.max_num_objects,
            'num_classes': graph_helper.num_classes,
            'num_states': graph_helper.num_states

        }

        self.args = args
        num_actions = graph_helper.num_actions
        if args.agent_type == 'hrl_mcts':
            self.objects1, self.objects2 = [], []
            # all objects that can be grabbed
            grabbed_obj = graph_helper.object_dict_types["objects_grab"]
            # all objects that can be opene
            container_obj = graph_helper.object_dict_types["objects_inside"]
            surface_obj = graph_helper.object_dict_types["objects_surface"]
            for act in ['open', 'pickplace']:
                if act == 'open':
                    #self.actions.append('open')
                    self.objects1.append("None")
                    self.objects2 += container_obj
                if act == 'pickplace':
                    #self.actions.append('pickplace')
                    self.objects2 += container_obj + surface_obj
                    self.objects1 += grabbed_obj

            self.objects1 = list(set(self.objects1))
            self.objects2 = list(set(self.objects2))

            # Goal locations
            # self.objects1 = ["cupcake", "apple"]
            self.objects2 = ["coffeetable", "kitchentable", "dishwasher", "fridge"]
            action_space = spaces.Tuple((
                spaces.Discrete(len(self.objects1)),
                spaces.Discrete(len(self.objects2))
            ))

        else:
            action_space = spaces.Tuple((spaces.Discrete(num_actions),
                                         spaces.Discrete(args.max_num_objects)))

        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.actor_critic_low_level = None
        self.actor_critic_low_level_put = None
        # ipdb.set_trace()
        if self.args.num_processes == 1:

            self.rl_agent_id = [ag_id for ag_id, agent in enumerate(self.arenas[0].agents) if 'RL' in agent.agent_type][
                0]
            # ipdb.set_trace()
            self.actor_critic = self.arenas[0].agents[self.rl_agent_id].actor_critic
            if self.arenas[0].agents[self.rl_agent_id].agent_type == 'RL_MCTS_RL':
                self.actor_critic_low_level = self.arenas[0].agents[self.rl_agent_id].actor_critic_low_level
                self.actor_critic_low_level_put = self.arenas[0].agents[self.rl_agent_id].actor_critic_low_level_put

        else:
            if args.use_alice:
                self.rl_agent_id = 1
            else:
                self.rl_agent_id = 0
            if args.agent_type == 'hrl_mcts':
                self.actor_critic = actor_critic_hl_mcts.ActorCritic(action_space, base_name=args.base_net,
                                                                     base_kwargs=base_kwargs)
                self.actor_critic.base.main.main.bad_transformer = False

            else:
                self.actor_critic = actor_critic.ActorCritic(action_space, base_name=args.base_net, base_kwargs=base_kwargs)

        self.actor_critic.to(self.device)

        self.memory_capacity_episodes = args.memory_capacity_episodes
        self.memory_all = []
        self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=args.lr)

        self.logger = None
        if args.logging:
            self.logger = Logger(args)

        atexit.register(self.close)

    def close(self):
        print('closing')
        del(self.arenas[0])

    def rollout(self, logging_value=0, record=False, episode_id=None, train=True, goals=None):
        """ Reward for an episode, get the cum reward """

        # Reset hidden state of agents
        # TODO: uncomment
        if self.args.num_processes == 1:
            info_envs = [self.arenas[0].rollout(logging_value, record, episode_id=episode_id, is_train=train, goals=goals)]
        else:
            async_routs = []
            for arena_id, arena in enumerate(self.arenas):
                curr_log = 0 if arena_id > 0 else logging_value
                async_routs.append(arena.rollout_reset.remote(curr_log, record, episode_id=episode_id, is_train=train, goals=goals))

            info_envs = []
            for async_rout in async_routs:
                info_envs.append(ray.get(async_rout))

        # # TODO: comment
        # info_envs = []
        # info_envs.append(self.arenas[0].rollout(logging, record))

        rewards = []
        info_rollout = []
        for process_data in info_envs:
            rewards.append(process_data[0])
            # successes.append(process_data[1]['success'])
            rollout_memory = process_data[2]

            # only get info form one process
            info_rollout.append(process_data[1])


            # Add into memory
            if train:
                for mem in rollout_memory[self.rl_agent_id]:

                    self.memory_all.append(*mem)
                self.memory_all.append(self.memory_all.goal[self.memory_all.position], None, None, None, 0, 0, 0)

        return rewards, info_rollout

    def load_model(self, model_path, model_path_lowlevel=None):
        model = torch.load(model_path)[0]
        # ipdb.set_trace()
        # ipdb.set_trace()
        self.actor_critic.load_state_dict(model.state_dict())
        if self.actor_critic_low_level is not None and model_path_lowlevel is not None:
            
            torch.nn.Module.dump_patches = True
            model_low_level = torch.load(model_path_lowlevel)

            # ipdb.set_trace()
            # ipdb.set_trace()
            self.actor_critic_low_level.load_state_dict(model_low_level[0].state_dict())
            self.actor_critic_low_level_put.load_state_dict(model_low_level[1].state_dict())

    def eval(self, episode_id, goals=None):
        self.actor_critic.eval()
        for agent in self.arenas[0].agents:
            if 'RL' in agent.agent_type:
                agent.epsilon = 0.
        with torch.no_grad():
            c_r_all, info_rollout = self.rollout(episode_id=episode_id, logging_value=2, train=False, goals=goals)
        return c_r_all, info_rollout


    def train(self, trainable_agents=None):
        # pdb.set_trace()
        if len(self.args.load_model):
            self.load_model(self.args.load_model)
        self.memory_all = MemoryMask(self.memory_capacity_episodes)
        self.memory_all.reset()

        start_episode_id = 1
        start_time = time.time()
        total_num_steps = 0
        info_ep = []
        for episode_id in range(start_episode_id, self.args.nb_episodes):
            eps = utils_models.get_epsilon(self.args.init_epsilon, 
                                           self.args.final_epsilon,
                                           self.args.max_exp_episodes,
                                           episode_id)

            # eps = 0.
            time_prerout = time.time()

            # TODO: Uncomment
            if self.args.num_processes > 1:
                curr_model = self.actor_critic.state_dict()
                for k, v in curr_model.items():
                   curr_model[k] = v.cpu()
                m_id = ray.put(curr_model)
                # TODO: Uncomment
                ray.get([arena.set_weigths.remote(eps, m_id) for arena in self.arenas])
            else:
                for agent in self.arenas[0].agents:
                    if 'RL' in agent.agent_type:
                        agent.epsilon = eps

            logging_value = 0
            if episode_id % self.args.log_interval == 0:
                logging_value = 1
                if episode_id % self.args.long_log == 0:
                    logging_value = 2
            c_r_all, info_rollout = self.rollout(logging_value=logging_value)
            # pdb.set_trace()



            end_time = time.time()

            action_space = []
            obs_space = []
            successes = []
            num_steps = []
            episode_rewards = [c_r_all_roll[0] for c_r_all_roll in c_r_all]


            for info_rollout_ep in info_rollout:
                num_steps.append(info_rollout_ep['nsteps'])
                action_space.append(info_rollout_ep['action_space'])
                obs_space.append(info_rollout_ep['observation_space'])
                successes.append(info_rollout_ep['success'])

            total_num_steps += np.sum(num_steps)
            fps = total_num_steps*1.0/(end_time-start_time)
            print("episode: #{} steps: {} reward: {} finished: {}/{} FPS {} #Objects {} #Objects actions {}".format(
                episode_id, np.mean(num_steps),
                np.mean(episode_rewards),
                np.sum(successes), len(episode_rewards),
                fps, np.mean(obs_space), np.mean(action_space)))

            if episode_id % self.args.log_interval == 0:

                print(info_rollout_ep['goals'])
                # Auxiliary task
                if 'pred_close' in info_rollout[0].keys() and \
                        len(info_rollout[0]['pred_close']) > 0:

                    pred_close = torch.cat(info_rollout[0]['pred_close'], 0)
                    gt_close = torch.cat(info_rollout[0]['gt_close'], 0)
                    pred_goal = torch.cat(info_rollout[0]['pred_goal'], 0)
                    gt_goal = torch.cat(info_rollout[0]['gt_goal'], 0)
                    mask_nodes = torch.cat(info_rollout[0]['mask_nodes'], 0)

                if episode_id % self.args.long_log == 0:
                    # print("Target:")
                    # print(info_rollout[0]['target'][1])
                    script_done = info_rollout[0]['script']
                    script_tried = info_rollout[0]['action_tried']
                    for iti, (script_t, script_d) in enumerate(zip(script_tried, script_done)):
                        info_step = ''
                        for relation in ['CLOSE', 'INSIDE', 'ON']:
                            if relation == 'INSIDE':
                                if len([x for x in info_rollout[0]['step_info'][iti][1] if x[2] == relation]) == 0:
                                    pdb.set_trace()

                            info_step += '  {}:  {}'.format(relation, ' '.join(
                                ['{}.{}'.format(x[0], x[1]) for x in info_rollout[0]['step_info'][iti][1] if x[2] == relation]))

                        if script_d is None:
                            script_d = ''

                        if False: #info_rollout[0]['step_info'][iti][0] is not None:
                            char_info = '{:07.3f} {:07.3f}'.format(info_rollout[0]['step_info'][iti][0]['center'][0],
                                                                   info_rollout[0]['step_info'][iti][0]['center'][2])
                            print('{: <36} --> {: <36} | char: {}  {}'.format(script_t, script_d, char_info, info_step))
                        else:
                            print('{: <36} --> {: <36}'.format(script_t, script_d))

                    if self.logger:

                        info_episode = {
                            'episode': episode_id,
                            'success': successes[0],
                            'reward': episode_rewards[0],
                            'script_tried': info_rollout[0]['action_tried'],
                            'script_done': info_rollout[0]['script'],
                            'target': info_rollout[0]['target'],
                            'info_step': info_rollout[0]['step_info'],
                            'graph': info_rollout[0]['graph'],
                            'visible_ids': info_rollout[0]['visible_ids'],
                            'action_ids': info_rollout[0]['action_space_ids'],
                        }
                        if 'pred_close' in info_rollout[0].keys():
                            info_episode['pred_close'] = info_rollout[0]['pred_close']
                        self.logger.log_info(info_episode)


            if self.logger:
                if episode_id % self.args.log_interval == 0:
                    epsilon = info_rollout[0]['epsilon']

                    dist_entropy = (np.mean([np.mean(info_rollout[it]['entropy'][0]) for it in range(len(info_rollout))]),
                                    np.mean([np.mean(info_rollout[it]['entropy'][1]) for it in range(len(info_rollout))]))
                    # pdb.set_trace()
                    info_aux = {}

                    if 'pred_close' in info_rollout[0].keys() and \
                            len(info_rollout[0]['pred_close']) > 0:
                        pred_closem = (pred_close.squeeze(-1) > 0.5).float().cpu()
                        tp = (mask_nodes.float() * gt_close.float() * pred_closem.float()).sum()
                        p = (mask_nodes.float() * gt_close.float()).sum()
                        fp = (mask_nodes.float() * (1. - gt_close.float()) * pred_closem.float()).sum()

                        info_aux['accuracy_goal'] = (gt_goal.cpu() == pred_goal.argmax(1).cpu()).float().mean().numpy()
                        info_aux['precision_close'] = (tp/(1e-9 + tp + fp)).numpy()
                        info_aux['recall_close'] = (tp/(1e-9 + p)).numpy()
                        info_aux['loss_close'] = nn.functional.binary_cross_entropy_with_logits(pred_close.squeeze(-1).cpu(),
                                                                                                gt_close,
                                                                                                mask_nodes).detach().numpy()
                        info_aux['loss_goal'] = nn.functional.cross_entropy(pred_goal.cpu(), gt_goal).detach().numpy()
                    self.logger.log_data(episode_id, episode_id, fps, episode_rewards,
                                         dist_entropy, epsilon, successes, num_steps, info_aux)



            # ===================== off-policy training =====================
            if not self.args.on_policy and episode_id - start_episode_id + 1 >= self.args.replay_start:
                nb_replays = 1
                for replay_id in range(nb_replays):
                    if self.args.balanced_sample:
                        trajs = self.memory_all.sample_batch_balanced_multitask(
                            self.args.batch_size,
                            self.args.neg_ratio,
                            maxlen=self.args.max_number_steps,
                            cutoff_positive=5.0)
                    else:
                        trajs = self.memory_all.sample_batch(
                            self.args.batch_size,
                            maxlen=self.args.max_number_steps)
                    N = len(trajs[0])
                    policies, actions, rewards, Vs, old_policies, dones, masks, loss_closes, loss_goals, nsteps_s = \
                        [], [], [], [], [], [], [], [], [], []

                    hx = torch.zeros(N, self.actor_critic.hidden_size).to(self.device)
                    cx = torch.zeros(N, self.actor_critic.hidden_size).to(self.device)

                    state_keys = trajs[0][0].state.keys()
                    print("Length trajectory: ", len(trajs))
                    for t in range(len(trajs) - 1):

                        # TODO: decompose here
                        inputs = {state_key: torch.cat([trajs[t][i].state[state_key] for i in range(N)]).to(self.device) for state_key in state_keys}

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
                        nsteps = np.array([trajs[t][i].nsteps for i in range(N)]).reshape((N, 1))
                        # policy, v, (hx, cx) = self.agents[agent_id].act(inputs, hx, mask)
                        v, _, policy, (hx, cx), out_dict = self.actor_critic.act(inputs, (hx, cx), mask, action_indices=action)

                        if hasattr(self.actor_critic, 'auxiliary_pred'):
                            auxiliary_out = self.actor_critic.auxiliary_pred((out_dict))
                        else:
                            auxiliary_out = {}
                        if 'pred_goal' in auxiliary_out:
                            pred_goal, pred_close = auxiliary_out['pred_goal'], auxiliary_out['pred_close']
                            gt_close = inputs['gt_close']
                            gt_goal = inputs['gt_goal']
                            mask_nodes = inputs['mask_object']
                            loss_close = nn.functional.binary_cross_entropy_with_logits(pred_close.squeeze(-1), gt_close, mask_nodes)
                            loss_goal = nn.functional.cross_entropy(pred_goal, gt_goal)
                        else:
                            loss_close = None
                            loss_goal = None

                        [array.append(element) for array, element in
                         zip((policies, actions, rewards, Vs, old_policies, dones, masks, nsteps_s, loss_closes, loss_goals),
                             (policy, action, reward, v, old_policy, done, mask, nsteps, loss_close, loss_goal))]


                        dones.append(done)

                        if (t + 1) % self.args.t_max == 0:  # maximum bptt length
                            hx = hx.detach()
                            cx = cx.detach()

                    self._train(self.actor_critic,
                                self.optimizer,
                                policies,
                                Vs,
                                actions,
                                rewards,
                                dones,
                                masks,
                                loss_closes,
                                loss_goals,
                                old_policies,
                                nsteps_s,
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
               loss_closes,
               loss_goals,
               old_policies,
               nsteps_s,
               verbose=0):
        """training"""

        off_policy = old_policies is not None
        policy_loss, value_loss, entropy_loss, loss_close, loss_goal = 0, 0, 0, 0, 0
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
            # pdb.set_trace()
            Vret = torch.from_numpy(rewards[i]).float() + torch.tensor(np.power(args.gamma, nsteps_s[i])) * Vret
            A = Vret.to(self.device) - Vs[i]

            log_prob_action = policies[i][0].gather(1, actions[i][0]).log()
            log_prob_object = policies[i][1].gather(1, actions[i][1]).log()
            log_prob = log_prob_action + log_prob_object

            if loss_closes[i] is not None:
                loss_close += loss_closes[i]
                loss_goal += loss_goals[i]

            #print(log_prob_action, log_prob_object)
            if off_policy:
                prob = policies[i][0].gather(1, actions[i][0]).data * policies[i][1].gather(1, actions[i][1]).data

                prob_action_old = old_policies[i][0].gather(1, actions[i][0]).data
                prob_object_old = old_policies[i][1].gather(1, actions[i][1]).data
                prob_old = prob_action_old * prob_object_old + 1e-6

                # rho = torch.exp(log_prob.data - log_prob_old.data).clamp(max=10.0)
                rho = (prob / prob_old).clamp(max=10.0)
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
            loss_goal /= episode_length
            loss_close /= episode_length

        if torch.isnan(policy_loss).any():
            pdb.set_trace()

        if verbose:
            print("policy_loss:", policy_loss.data.cpu().numpy()[0])
            print("value_loss:", value_loss.data.cpu().numpy()[0])
            print("entropy_loss:", entropy_loss.data.cpu().numpy())

            if loss_goal != 0.:
                print("loss_goal:", loss_goal.data.cpu().numpy())
                print("loss_close:", loss_close.data.cpu().numpy())

        # updating net
        optimizer.zero_grad()
        loss = policy_loss + value_loss + entropy_loss * args.entropy_coef
        loss = loss + loss_close * args.c_loss_close + loss_goal * args.c_loss_goal
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm, 1)
        optimizer.step()
