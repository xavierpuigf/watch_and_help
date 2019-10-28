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

def test(dataset, helper, policy_net):

    # Loading params
    policy_net.eval()

    envs = []
    for dp in dataset:
        graph_file, goal, program = dp
        curr_env = Environment(graph_file, goal, 2)
        for agent in curr_env.agents:
            agent.policy_net = policy_net
            agent.activation_info = policy_net.activation_info()
        envs.append(curr_env)

    num_rollouts = helper.args.num_rollouts

    for it, dp in enumerate(dataset):
        graph_file, program, goal = dp
        curr_env = envs[it]
        agent = curr_env.agents[0]
        instructions = []
        for it2 in range(num_rollouts):
            # Decide object
            observations = agent.get_observations()
            instr_info = agent.get_instruction(observations)

            instr = instr_info['instruction']
            instructions.append(instr)
            if instr == '[stop]':
                break
            r, states, infos = curr_env.env.step(instr)

        print('PRED:')
        print('\n'.join(instructions))

        print('GT')
        print('\n'.join(program))
        pdb.set_trace()
        lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS(program, instructions)
        print('LCSaction {:.2f}. LCSo1 {:.2f}. LCSo2 {:.2f}. LCStriplet {:.2f}'.format(
            lcs_action, lcs_o1, lcs_o2, lcs_triple))


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

    data_loader = data.DataLoader(dataset, batch_size=helper.args.batch_size, shuffle=do_shuffle, num_workers=0)
    for epoch in range(num_epochs):
        for it, dp in enumerate(data_loader):

            state, program, goal = dp
            action_logits, o1_logits, o2_logits, repr = policy_net(state, goal)
            logits = action_logits, o1_logits, o2_logits

            #action_logits[0,0,:].sum().backward()
            #pdb.set_trace()
            loss, aloss, o1loss, o2loss, debug = bc_loss(program, logits)

            # Obtain the prediction
            pred_action = torch.argmax(action_logits, -1)
            pred_o1 = torch.argmax(o1_logits, -1)
            pred_o2 = torch.argmax(o2_logits, -1)

            object_ids = state[1]
            object_names = state[0]
            object_names_pred_1 = object_names[np.arange(object_names.shape[0])[:, None],
                                               np.arange(object_names.shape[1])[None, :], pred_o1]
            object_ids_pred_1 = object_ids[np.arange(object_names.shape[0])[:, None],
                                           np.arange(object_names.shape[1])[None, :], pred_o1]
            object_names_pred_2 = object_names[np.arange(object_names.shape[0])[:, None],
                                               np.arange(object_names.shape[1])[None, :], pred_o2]
            object_ids_pred_2 = object_ids[np.arange(object_names.shape[0])[:, None],
                                           np.arange(object_names.shape[1])[None, :], pred_o2]

            object_names_gt_1 = object_names[np.arange(object_names.shape[0])[:, None],
                                             np.arange(object_names.shape[1])[None, :], program[1]]
            object_ids_gt_1 = object_ids[np.arange(object_names.shape[0])[:, None],
                                         np.arange(object_names.shape[1])[None, :], program[1]]
            object_names_gt_2 = object_names[np.arange(object_names.shape[0])[:, None],
                                             np.arange(object_names.shape[1])[None, :], program[2]]
            object_ids_gt_2 = object_ids[np.arange(object_names.shape[0])[:, None],
                                         np.arange(object_names.shape[1])[None, :], program[2]]

            pred_instr = obtain_list_instr(pred_action, object_names_pred_1, object_ids_pred_1,
                                           object_names_pred_2, object_ids_pred_2, dataset)
            gt_instr = obtain_list_instr(program[0], object_names_gt_1, object_ids_gt_1,
                                         object_names_gt_2, object_ids_gt_2, dataset)

            # agent.reset()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % helper.args.print_freq == 0:
            lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS_multiple(gt_instr, pred_instr)
            print(utils.pretty_print_program(pred_instr[0]))
            print('Epoch:{}. Iter {}.  Loss {:.3f}. ActionLoss {:.3f}. O1Loss {:.3f}. O2Loss {:.3f}.'
                  'ActionLCS {:.2f}. O1LCS {:.2f}. O2LCS {:.2f}. TripletLCS {:.2f}'.format(
                epoch, it,
                loss.data, aloss.data, o1loss.data, o2loss.data,
                lcs_action, lcs_o1, lcs_o2, lcs_triple))

    test(dataset, helper, policy_net)
    #pdb.set_trace()


def obtain_list_instr(actions, o1_names, o1_ids, o2_names, o2_ids, dataset):
    # Split by batch
    actions = torch.unbind(actions.cpu().data, 0)
    o1_names = torch.unbind(o1_names.cpu().data, 0)
    o1_ids = torch.unbind(o1_ids.cpu().data, 0)
    o2_names = torch.unbind(o2_names.cpu().data, 0)
    o2_ids = torch.unbind(o2_ids.cpu().data, 0)

    num_batches = len(actions)
    programs = []
    for it in range(num_batches):
        o1 = zip(list(o1_names[it].numpy()), list(o1_ids[it].numpy()))
        o2 = zip(list(o2_names[it].numpy()), list(o2_ids[it].numpy()))

        action_list = [dataset.action_dict.get_el(x) for x in list(actions[it].numpy())]
        object_1_list = [(dataset.object_dict.get_el(x), idi) for x, idi in o1]
        object_2_list = [(dataset.object_dict.get_el(x), idi) for x, idi in o2]


        programs.append((action_list, object_1_list, object_2_list))
    return programs


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
    train(dataset, helper)

    pdb.set_trace()

if __name__ == '__main__':
    start()