import argparse
import random
import time
import os
import json
import numpy as np
import pdb

import torch
from torch.utils.tensorboard import SummaryWriter
from helper import to_cpu, average_over_list, writer_helper


def grab_args():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--train_iters', type=int, default=1e7)
    parser.add_argument('--inputtype', type=str, default='actioninput')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--inference', type=int, default=0)
    parser.add_argument('--single', type=int, default=0)

    # model config
    parser.add_argument(
        '--model_type',
        type=str,
        default='max')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--predicate_hidden', type=int, default=128)
    parser.add_argument('--demo_hidden', type=int, default=128)
    parser.add_argument('--multi_classifier', type=int, default=0)
    parser.add_argument('--transformer_nhead', type=int, default=2)
    

    # train config
    parser.add_argument(
        '--gpu_id',
        metavar='N',
        type=str,
        nargs='+',
        help='specify the gpu id')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_lr_rate', type=float, default=3e-4)

    args = parser.parse_args()
    return args


def setup(train):

    def _basic_setting(args):

        # set seed
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.gpu_id is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            args.__dict__.update({'cuda': False})
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(args.gpu_id)
            args.__dict__.update({'cuda': True})
            torch.cuda.manual_seed_all(args.seed)

        if args.debug:
            args.verbose = True

    def _basic_checking(args):
        pass

    def _create_checkpoint_dir(args):

         # setup checkpoint_dir
        if args.debug:
            checkpoint_dir = 'debug'
        elif train:
            checkpoint_dir = 'checkpoint_dir'
        else:
            checkpoint_dir = 'testing_dir'

        checkpoint_dir = os.path.join(checkpoint_dir, 'demo2predicate')

        args_dict = args.__dict__
        keys = sorted(args_dict)
        prefix = ['{}-{}'.format(k, args_dict[k]) for k in keys]
        prefix.remove('debug-{}'.format(args.debug))
        prefix.remove('checkpoint-{}'.format(args.checkpoint))
        prefix.remove('gpu_id-{}'.format(args.gpu_id))

        checkpoint_dir = os.path.join(checkpoint_dir, *prefix)

        checkpoint_dir += '/{}'.format(time.strftime("%Y%m%d-%H%M%S"))

        return checkpoint_dir

    def _make_dirs(checkpoint_dir, tfboard_dir):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(tfboard_dir):
            os.makedirs(tfboard_dir)

    def _print_args(args):

        args_str = ''
        with open(os.path.join(checkpoint_dir, 'args.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                s = '{}: {}'.format(k, v)
                args_str += '{}\n'.format(s)
                print(s)
                f.write(s + '\n')
        print("All the data will be saved in", checkpoint_dir)
        return args_str

    args = grab_args()
    _basic_setting(args)
    _basic_checking(args)

    checkpoint_dir = args.checkpoint
    tfboard_dir = os.path.join(checkpoint_dir, 'tfboard')

    _make_dirs(checkpoint_dir, tfboard_dir)
    args_str = _print_args(args)

    writer = SummaryWriter(tfboard_dir)
    writer.add_text('args', args_str, 0)
    writer = writer_helper(writer)

    model_config = {
        "model_type": args.model_type,
        "embedding_dim": args.embedding_dim,
        "predicate_hidden": args.predicate_hidden,
    }
    model_config.update({"demo_hidden": args.demo_hidden})

    return args, checkpoint_dir, writer, model_config


def summary(
        args,
        writer,
        info,
        train_args,
        model,
        test_loader,
        postfix, 
        fps=None):

    
    if postfix == 'train':
        model.write_summary(writer, info, postfix=postfix)
    elif postfix == 'val':
        info = summary_eval(
            model,
            test_loader,
            test_loader.dataset)
        model.write_summary(writer, info, postfix=postfix)
    elif postfix == 'test':
        info = summary_eval(
            model,
            test_loader,
            test_loader.dataset)

        model.write_summary(writer, info, postfix=postfix)
    else:
        raise ValueError

    if fps:
        writer.scalar_summary('General/fps', fps)

    return info


def summary_eval(
        model,
        loader,
        dset):

    model.eval()
    with torch.no_grad():

        loss_list = []
        top1_list = []
        iter = 0
        
        prob = []
        target = []
        file_name = []
        for batch_data in loader:
            loss, info = model(batch_data)
            loss_list.append(loss.cpu().item())
            top1_list.append(info['top1'])

            prob.append(info['prob'])
            target.append(info['target'])
            file_name.append(info['file_name'])

            if iter%10==0:
                print('testing %d / %d: loss %.4f: acc %.4f' % (iter, len(loader), loss, info['top1']))

            iter += 1

    info = {"loss": sum(loss_list)/ len(loss_list), "top1": sum(top1_list)/ len(top1_list), "prob": prob, "target": target, "file_name": file_name}
    return info




def save(args, i, checkpoint_dir, model):
    
    # save_path = '{}/demo2predicate-{}.ckpt'.format(checkpoint_dir, i)    
    save_path = '{}/demo2predicate-{}.ckpt'.format(checkpoint_dir, 'best_model')
    model.save(save_path, True)

