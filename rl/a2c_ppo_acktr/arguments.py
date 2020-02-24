import argparse

import torch
import pdb

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
            '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--task_type', default='find', choices=['find', 'complex'], help='algorithm to use: find | complex')

    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
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
        default='python',
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
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--model_type',
        default='CNN',
        choices=['GNN', 'CNN'],
        help='use a linear schedule on the learning rate')


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
