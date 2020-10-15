import resource
import time
from termcolor import colored
import torch
from torch.utils.data import DataLoader
from helper import Constant, LinearStep
from predicate.utils import save, setup
from predicate.utils import summary
import ipdb
import pdb
import random
import json
import pickle

topk = 1
p_th = 0.5


def print_output(args, outputs, targets, file_names, test_dset):

    goal_predicates = test_dset.goal_predicates
    goal_predicates = {v:k for k,v in goal_predicates.items()}

    json_output = {}

    for i, target in enumerate(targets):
        
        if args.inference == 0:
            p = random.uniform(0, 1)
            if p>p_th:
                continue

        file_name = file_names[i]
        output = outputs[i]

        if args.multi_classifier:
            output = torch.Tensor(output).view(-1, len(test_dset.goal_predicates), test_dset.max_subgoal_length+1)
            target = torch.Tensor(target).view(-1, len(test_dset.goal_predicates))
        else:
            output = torch.Tensor(output).view(-1, test_dset.max_goal_length, len(test_dset.goal_predicates))
            target = torch.Tensor(target).view(-1, test_dset.max_goal_length)

        output = output.numpy()
        target = target.numpy()

        if args.inference == 0:
            target_inference = [target[0]]
            output_inference = [output[0]]
            file_name_inference = [file_name[0]]
        else:
            target_inference = target
            output_inference = output
            file_name_inference = file_name


        for (target_j, output_j, file_name_j) in zip(target_inference, output_inference, file_name_inference):
            ## only show the fist sample in each minibatch
            assert file_name_j not in json_output
            json_output[file_name_j] = {}
            json_output[file_name_j]['ground_truth'] = []
            json_output[file_name_j]['prediction'] = []
            json_output[file_name_j]['ground_truth_id'] = []
            json_output[file_name_j]['prediction_id'] = []

            print('----------------------------------------------------------------------------------')

            
            if args.multi_classifier:
                assert len(target_j) == len(goal_predicates) == len(output_j)
                for k, target_k in enumerate(target_j):
                    output_k = output_j[k]
                    strtar = ('tar: %s   %d' % (goal_predicates[k], target_k)).ljust(50, ' ')
                    strpre = '| gen: %s   %d' % (goal_predicates[k], output_k.argmax())
                    print(strtar+strpre)

                    json_output[file_name_j]['ground_truth_id'].append(int(target_k))
                    json_output[file_name_j]['prediction_id'].append(output_k.argmax())
                    json_output[file_name_j]['ground_truth'].append(goal_predicates[k])
                    json_output[file_name_j]['prediction'].append(goal_predicates[k])
            else:
                for k, target_k in enumerate(target_j):
                    output_k = output_j[k]

                    strtar = ('tar: %s' % goal_predicates[int(target_k)]).ljust(50, ' ')
                    strpre = '| gen: %s' % goal_predicates[output_k.argmax()]
                    print(strtar+strpre)

                    json_output[file_name_j]['ground_truth_id'].append(int(target_k))
                    json_output[file_name_j]['prediction_id'].append(output_k.argmax())
                    json_output[file_name_j]['ground_truth'].append(goal_predicates[int(target_k)])
                    json_output[file_name_j]['prediction'].append(goal_predicates[output_k.argmax()])


            # for j, target_j in enumerate(target):
            #     output_j = output[j]
            #     print('tar: %s ||| gen: %s ' % (goal_predicates[target_j], goal_predicates[output_j.argmax()]))
            print('----------------------------------------------------------------------------------')


    if args.inference == 1:
        if args.single:
            pickle.dump( json_output, open( "dataset/test_output_"+args.resume.split('/')[-2]+"_single_task.p", "wb" ) )
        else:
            pickle.dump( json_output, open( "dataset/test_output_"+args.resume.split('/')[-2]+"_multiple_task.p", "wb" ) )




def run_one_iteration(model, optim, batch_data, train_args, args):
    model.train()
    optim.zero_grad()
    loss, info = model(batch_data, **train_args)
    loss.backward()

    # torch.nn.utils.clip_grad_norm(model.parameters(), 0.01)

    optim.step()

    return batch_data, info, loss


def train(
        args,
        model,
        optim,
        train_loader,
        test_loader,
        val_loader,
        checkpoint_dir,
        writer,
        train_dset,
        test_dset):

    # Train
    print(colored('Start training...', 'red'))
    # loader for the testing set
    def _loader():
        while True:
            for batch_data in test_loader:
                yield batch_data
    get_next_data_fn = _loader().__iter__().__next__

    train_args = {}
    

    if args.inference == 1:
        info = summary(
            args,
            writer,
            None,
            None,
            model,
            test_loader,
            'test')
        
        print_output(args, info['prob'], info['target'], info['file_name'], test_dset)
        print('test top1', info['top1'])
        
        return 0


    def _train_loop():

        iter = 0
        summary_t1 = time.time()

        test_best_top1 = 0
        while iter <= args.train_iters:
            for batch_data in train_loader:
                results = run_one_iteration(
                    model, optim, batch_data, train_args, args)
                batch_data, info, loss = results

                if iter % 10 == 0:
                    print('%s: training %d / %d: loss %.4f: acc %.4f' % (args.checkpoint, iter, len(train_loader), loss, info['top1']))

                    fps = 10. / (time.time() - summary_t1)
                    info = summary(
                        args,
                        writer,
                        info,
                        train_args,
                        model,
                        None,
                        'train',
                        fps=fps)
                    if iter > 0:
                        summary_t1 = time.time()
                
                if iter % (len(train_loader)*5) == 0 and iter>0:
                    info = summary(
                        args,
                        writer,
                        None,
                        None,
                        model,
                        test_loader,
                        'test')

                    if info['top1']>test_best_top1:
                        test_best_top1 = info['top1']
                        save(args, iter, checkpoint_dir, model)

                    print_output(args, info['prob'], info['target'], info['file_name'], test_dset)

                iter += 1
    
    _train_loop()


def main():
    
    args, checkpoint_dir, writer, model_config = setup(train=True)
    
    print(args)

    from predicate.demo_dataset_graph import get_dataset
    from predicate.demo_dataset_graph import collate_fn
    from predicate.demo_dataset_graph import to_cuda_fn
    train_dset, test_dset, new_test_dset = get_dataset(args, train=True)


    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        drop_last=True)


    if args.single:
        test_loader = DataLoader(
            dataset=test_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False)

        val_loader = DataLoader(
            dataset=test_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False)
    else:
        test_loader = DataLoader(
            dataset=new_test_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False)

        val_loader = DataLoader(
            dataset=new_test_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False)

    # initialize model
    if args.inputtype=='graphinput':
        from network.encoder_decoder import GraphDemo2Predicate
        model = GraphDemo2Predicate(args, train_dset, **model_config)
    elif args.inputtype=='actioninput':
        from network.encoder_decoder import ActionDemo2Predicate
        model = ActionDemo2Predicate(args, train_dset, **model_config)


    if args.resume!='':
        model.load(args.resume, True)


    optim = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        args.model_lr_rate)
    if args.gpu_id is not None:
        model.cuda()
        model.set_to_cuda_fn(to_cuda_fn)


    
    # main loop
    train(
        args,
        model,
        optim,
        train_loader,
        test_loader,
        val_loader,
        checkpoint_dir,
        writer,
        train_dset, 
        test_dset)


    ## final inference
    if args.inference != 1:
        args.inference = 1
        train(
            args,
            model,
            optim,
            train_loader,
            test_loader,
            val_loader,
            checkpoint_dir,
            writer,
            train_dset, 
            test_dset)


# See: https://github.com/pytorch/pytorch/issues/973
# the worker might quit loading since it takes too long
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024 * 4, rlimit[1]))


if __name__ == '__main__':
    # See: https://github.com/pytorch/pytorch/issues/1355
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
