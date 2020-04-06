import glob
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import json
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None




# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)



def get_epsilon(init_eps, end_eps, num_steps, episode):
    return (init_eps - end_eps) * (1.0 - min(1.0, float(episode) / float(num_steps))) + end_eps

class AggregatedStats():
    def __init__(self):
        self.success_per_apt = {}
        self.success_per_goal = {}

    def reset(self):
        success_per_apt = {}
        success_per_goal = {}

    def add(self, success, goal, apt):
        if goal not in self.success_per_goal:
            sg_count, g_count = 0, 0
        else:
            sg_count, g_count = self.success_per_goal[goal]
        if apt not in self.success_per_apt:
            sa_count, a_count = 0, 0
        else:
            sa_count, a_count = self.success_per_apt[apt]
        r = 1 if success else 0
        self.success_per_goal[goal] = [sg_count + r, g_count + 1]
        self.success_per_apt[apt] = [sa_count + r, a_count + 1]

    def add_list(self, info_list):
        for item_info in info_list:
            self.add(item_info['success'], item_info['goal'], item_info['apt'])

    def barplot(self, values, names=None, special_ids=None):
        fig, ax = plt.subplots()
        index = range(values.shape[0])

        a1 = plt.bar(index, values)

        if names is not None:
            ax.set_xticks(np.asarray([i for i in range(len(names))]))
            ax.set_xticklabels(names, rotation=65)
        if special_ids is not None:
            a1[special_ids].set_color('r')
        plt.tight_layout()
        return fig

    def create_histogram(self, cont_dict):
        keys = sorted(list(cont_dict.keys()))
        values = [cont_dict[x] for x in keys]
        success = np.array([x[0] for x in values])
        cont = np.array([x[1] for x in values])
        return self.barplot(success, keys), self.barplot(cont, keys)



    def print_hist(self, tb_writer):
        img_goal, cnt_goal = self.create_histogram(self.success_per_goal)
        img_apt, cnt_apt = self.create_histogram(self.success_per_apt)

        tb_writer.add_figure("histogram/success_per_goal", img_goal)
        tb_writer.add_figure("histogram/success_per_apt", img_apt)
        tb_writer.add_figure("histogram/count_per_goal", cnt_goal)
        tb_writer.add_figure("histogram/count_per_apt", cnt_apt)



class Logger():
    def __init__(self, args):
        self.args = args
        self.experiment_name = self.get_experiment_name()
        self.tensorboard_writer = None
        self.save_dir = args.save_dir

        now = datetime.datetime.now()
        self.tstmp = now.strftime('%Y-%m-%d_%H-%M-%S')
        self.set_tensorboard()
        self.first_log = False
        self.stats = AggregatedStats()

        save_path = os.path.join(self.save_dir, self.experiment_name)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        with open('{}/{}/args.txt'.format(self.save_dir, self.experiment_name), 'w+') as f:
            dict_args = vars(args)
            f.writelines(json.dumps(dict_args, indent=4))


    def set_tensorboard(self):
        now = datetime.datetime.now()
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.args.tensorboard_logdir,
                                                                     self.experiment_name, self.tstmp))

    def get_experiment_name(self):
        args = self.args
        experiment_name = 'env.{}/task.{}-numproc.{}-obstype.{}-sim.{}/taskset.{}/'\
                          'mode.{}-algo.{}-base.{}-gamma.{}-lr{}'.format(
            args.env_name,
            args.task_type,
            args.num_processes,
            args.obs_type,
            args.simulator_type,
            args.task_set,
            args.train_mode,
            args.algo,
            args.base_net,
            args.gamma,
            args.lr)

        if args.debug:
            experiment_name += 'debug'
        return experiment_name

    def log_data(self, j, total_num_steps, start, end, episode_rewards, dist_entropy, epsilon, successes, info_episodes):
        if self.first_log:
            self.first_log = False
            if self.args.tensorboard_logdir is not None:
                self.set_tensorboard()
        fps = total_num_steps / (end - start)
        # print(
        #     "Updates {}, num timesteps {}, FPS {} "
        #     "\n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        #         .format(j, total_num_steps,
        #                 int(total_num_steps / (end - start)),
        #                 len(episode_rewards), np.mean(episode_rewards),
        #                 np.median(episode_rewards), np.min(episode_rewards),
        #                 np.max(episode_rewards), dist_entropy, value_loss,
        #                 action_loss))
        # self.stats.add_list(info_episodes)
        # self.stats.print_hist(self.tensorboard_writer)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("sum_reward", np.sum(episode_rewards), total_num_steps)
            # tensorboard_writer.add_scalar("median_reward", np.median(episode_rewards), total_num_steps)
            # tensorboard_writer.add_scalar("min_reward", np.min(episode_rewards), total_num_steps)
            # tensorboard_writer.add_scalar("max_reward", np.max(episode_rewards), total_num_steps)
            self.tensorboard_writer.add_scalar("action_entropy/action", dist_entropy[0], total_num_steps)
            self.tensorboard_writer.add_scalar("action_entropy/object", dist_entropy[1], total_num_steps)
            # self.tensorboard_writer.add_scalar("losses/value_loss", value_loss, total_num_steps)
            # self.tensorboard_writer.add_scalar("losses/action_loss", action_loss, total_num_steps)
            self.tensorboard_writer.add_scalar("info/epsilon", epsilon, total_num_steps)
            self.tensorboard_writer.add_scalar("info/episode", j, total_num_steps)
            self.tensorboard_writer.add_scalar("info/success", successes, total_num_steps)
            self.tensorboard_writer.add_scalar("info/fps", fps, total_num_steps)

    def save_model(self, j, actor_critic):
        save_path = os.path.join(self.save_dir, self.experiment_name)

        torch.save([
            actor_critic,
        ], os.path.join(save_path, "{}.pt".format(j)))
