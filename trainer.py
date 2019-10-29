import json
import pdb
import torch
import utils
import numpy as np
import argparse
from envdataset import EnvDataset
from environment import Environment
from models.single_policy import SinglePolicy
from torch.utils import data

def test(dataset, data_loader, helper, policy_net, epoch):

    # Loading params

    policy_net.eval()

    with torch.no_grad():
        metrics = utils.AvgMetrics(['LCS', 'ActionLCS', 'O1LCS', 'O2LCS'], ':.2f')
        metrics_loss = utils.AvgMetrics(['Loss', 'ActionLoss', 'O1Loss', 'O2Loss'], ':.3f')

        metrics_loss.reset()
        metrics.reset()
        for it, dp in enumerate(data_loader):
            state, program, goal = dp
            action_logits, o1_logits, o2_logits, repr = policy_net(state, goal)
            logits = action_logits, o1_logits, o2_logits

            loss, aloss, o1loss, o2loss, debug = bc_loss(program, logits)

            # Obtain the prediction
            pred_action = torch.argmax(action_logits, -1)
            pred_o1 = torch.argmax(o1_logits, -1)
            pred_o2 = torch.argmax(o2_logits, -1)

            object_ids = state[1]
            object_names = state[0]
            pred_instr = utils.get_program_from_nodes(dataset, object_names, object_ids,
                                                      [pred_action, pred_o1, pred_o2])
            gt_instr = utils.get_program_from_nodes(dataset, object_names, object_ids, program)
            lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS_multiple(gt_instr, pred_instr)
            metrics_loss.update({
                'Loss': loss.data.cpu(),
                'ActionLoss': aloss.data.cpu(),
                'O1Loss': o1loss.data.cpu(),
                'O2Loss': o2loss.data.cpu()
            })
            metrics.update({'LCS': lcs_triple,
                            'ActionLCS': lcs_action,
                            'O1LCS': lcs_o1,
                            'O2LCS': lcs_o2})

    helper.log(epoch, metrics, 'LCS', 'test')
    helper.log(epoch, metrics_loss, 'Losses', 'test')


def train(dataset, helper):
    # Think how to handle this with multiple envs. One agent multiple envs?
    # Here we set the policy and the env
    policy_net = SinglePolicy(dataset, helper)
    policy_net.cuda()
    policy_net = torch.nn.DataParallel(policy_net)
    # policy_net = policy_net.cuda()
    # envs = []
    # for dp in dataset:
    #     state, program = dp
    #     actions, objects1, objects2 = policy_net(state)
    #     curr_env = Environment(graph_file, goal, 2)
    #     for agent in curr_env.agents:
    #         agent.policy_net = policy_net
    #         agent.activation_info = policy_net.activation_info()
    #     envs.append(curr_env)
    #
    optimizer = torch.optim.Adam(list(policy_net.parameters()))
    #
    # num_rollouts = helper.args.num_rollouts
    num_epochs = helper.args.num_epochs
    do_shuffle = not helper.args.debug

    if helper.args.debug:
        num_workers = 0
    else:
        num_workers = helper.args.num_workers

    data_loader = data.DataLoader(dataset, batch_size=helper.args.batch_size,
                                           shuffle=do_shuffle, num_workers=num_workers)
    data_loader_test = data.DataLoader(dataset, batch_size=helper.args.batch_size,
                                  shuffle=False, num_workers=num_workers)

    metrics = utils.AvgMetrics(['LCS', 'ActionLCS', 'O1LCS', 'O2LCS'], ':.2f')
    metrics_loss = utils.AvgMetrics(['Loss', 'ActionLoss', 'O1Loss', 'O2Loss'], ':.3f')

    for epoch in range(num_epochs):
        metrics_loss.reset()
        metrics.reset()
        for it, dp in enumerate(data_loader):

            state, program, goal = dp
            action_logits, o1_logits, o2_logits, repr = policy_net(state, goal)
            logits = action_logits, o1_logits, o2_logits

            loss, aloss, o1loss, o2loss, debug = bc_loss(program, logits)

            # Obtain the prediction
            pred_action = torch.argmax(action_logits, -1)
            pred_o1 = torch.argmax(o1_logits, -1)
            pred_o2 = torch.argmax(o2_logits, -1)

            object_ids = state[1]
            object_names = state[0]
            pred_instr = utils.get_program_from_nodes(dataset, object_names, object_ids, [pred_action, pred_o1, pred_o2])
            gt_instr = utils.get_program_from_nodes(dataset, object_names, object_ids, program)


            # agent.reset()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % helper.args.print_freq == 0:
                # pdb.set_trace()
                lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS_multiple(gt_instr, pred_instr)
                metrics.update({'LCS': lcs_triple,
                                'ActionLCS': lcs_action,
                                'O1LCS': lcs_o1,
                                'O2LCS': lcs_o2})
                metrics_loss.update({
                    'Loss': loss.data.cpu(),
                    'ActionLoss': aloss.data.cpu(),
                    'O1Loss': o1loss.data.cpu(),
                    'O2Loss': o2loss.data.cpu()
                })

                print(utils.pretty_print_program(pred_instr[0]))
                print('Epoch:{}. Iter {}.  Losses: {}'
                      'LCS: {}'.format(
                    epoch, it,
                    str(metrics_loss),
                    str(metrics)))

        helper.log(epoch, metrics, 'LCS', 'train')
        helper.log(epoch, metrics_loss, 'Losses', 'train')

        if (epoch + 1) % helper.args.save_freq == 0:
            helper.save(epoch, 0., policy_net.state_dict(), optimizer.state_dict())
        test(dataset, data_loader_test, helper, policy_net, epoch)
    #pdb.set_trace()





def bc_loss(program, logits):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    gt_a, gt_o1, gt_o2, mask = program
    mask = mask.float()
    if torch.cuda.is_available():
        mask = mask.cuda()
        gt_a = gt_a.cuda()
        gt_o1 = gt_o1.cuda()
        gt_o2 = gt_o2.cuda()
    #pdb.set_trace()
    l_a, l_o1, l_o2 = logits

    bs = gt_a.shape[0]
    loss_action = criterion(l_a.view(-1, l_a.shape[-1]), gt_a.view(-1)).view(bs, -1)
    loss_o1 = criterion(l_o1.view(-1, l_o1.shape[-1]), gt_o1.view(-1)).view(bs, -1)
    loss_o2 = criterion(l_o2.view(-1, l_o2.shape[-1]), gt_o2.view(-1)).view(bs, -1)
    m_loss_action = ((loss_action * mask).sum(1)/mask.sum(1)).mean()
    m_loss_o1 = ((loss_o1 * mask).sum(1) / mask.sum(1)).mean()
    m_loss_o2 = ((loss_o2 * mask).sum(1) / mask.sum(1)).mean()
    total_loss = m_loss_action + m_loss_o1 + m_loss_o2
    debug = {}
    debug['loss_o1'] = loss_o1
    debug['loss_o2'] = loss_o2
    return total_loss, m_loss_action, m_loss_o1, m_loss_o2, debug

def start():
    helper = utils.setup()
    dataset = EnvDataset(helper.args)
    dataset_test = EnvDataset(helper.args, 'test')
    train(dataset, helper)

    pdb.set_trace()

if __name__ == '__main__':
    start()