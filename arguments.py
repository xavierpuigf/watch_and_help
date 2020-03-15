import argparse

import torch
import pdb


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')
    parser.add_argument('--mode', type=str, default='full', choices=['simple', 'full'], help='Environment type')
    parser.add_argument('--num-per-apartment', type=int, default=3, help='Maximum #episodes/apartment')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--task_type', default='find', choices=['find', 'complex', 'open'], help='algorithm to use: find | complex')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient (default: 0.0)')
    parser.add_argument(
        '--init-epsilon',
        type=float,
        default=0.1,
        help='initial epsilon (default: 0.1)')
    parser.add_argument(
        '--final-epsilon',
        type=float,
        default=0.01,
        help='final epsilon (default: 0.01)')
    parser.add_argument(
        '--max-exp-episodes',
        type=int,
        default=10000,
        help='Maximum exploration episodes (default: 10000)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument(
        '--env-name',
        default='virtualhome',
        help='environment to train on (default: PongNoFrameskip-v4)')

    parser.add_argument(
        '--simulator-type',
        default='unity',
        choices=['unity', 'python'],
        help='whether to use unity or python sim')

    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')

    parser.add_argument(
        '--num-steps',
        type=int,
        default=100,
        help='number of forward steps in A2C (default: 100)')
    parser.add_argument(
        '--t-max',
        type=int,
        default=20,
        help='number of bptt steps (default: 20)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--log-dir',
        default='logs/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True,
        help='use a recurrent policy')

    parser.add_argument(
        '--num-frame-stack',
        type=int,
        default=1,
        help='number of frames to stack in the observations')

    parser.add_argument(
        '--obs_type',
        type=str,
        default='mcts',
        choices=['full', 'rgb', 'visibleid', 'mcts'],
    )

    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--attention_type',
        type=str,
        default='linear',
        choices=['fc', 'dot', 'linear'],
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--base_net',
        default='TF',
        choices=['GNN', 'CNN', 'TF'],
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--executable_file', type=str, default='../executables/exec_linux03.03/exec_linux03.3multiagent.x86_64')

    parser.add_argument(
        '--train_mode',
        default='RL',
        choices=['BC', 'RL']
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.05,
        help='epsilon for e-greedy')

    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='network dim')

    parser.add_argument(
        '--nb_episodes',
        type=int,
        default=2000,
        help='number of episodes')

    parser.add_argument(
        '--replay_start',
        type=int,
        default=2,
        help='when to start using the replay buffer')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size for replay buffer')

    parser.add_argument(
        '--t_max',
        type=int,
        default=100,
        help='number of steps until breaking bptt')

    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=200,
        help='number of episodes')

    parser.add_argument(
        '--balanced_sample',
        action='store_true',
        default=False,
        help='what')

    parser.add_argument(
        '--max-num-edges',
        type=int,
        default=300,
        help='how many objects in observation space')

    parser.add_argument(
        '--max-num-objects',
        type=int,
        default=150,
        help='how many objects in observation space')

    parser.add_argument(
        '--base-port', type=int, default=8080)

    parser.add_argument(
        '--display', type=str, default="2")

    parser.add_argument(
        '--max_gradient_norm', type=int, default=10)

    parser.add_argument(
        '--memory-capacity-episodes', type=int, default=10000)

    parser.add_argument('--no-time-normalization', action='store_true', default=False,
                        help='whether to run on or off policy')

    parser.add_argument('--on-policy', action='store_true', default=False,
                        help='whether to run on or off policy')

    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')

    parser.add_argument('--tensorboard-logdir', default='log_tb',
                        help='logs to tensorboard in the specified directory')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args


if __name__ == '__main__':
    args = get_args()