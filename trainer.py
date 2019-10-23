import json
import pdb
import torch
import utils
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
        graph_file, goal, program = dp
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

    data_loader = data.DataLoader(dataset)
    for epoch in range(num_epochs):
        for it, dp in enumerate(data_loader):

            state, program = dp
            action_logits, o1_logits, o2_logits = policy_net(state)
            logits = action_logits, o1_logits, o2_logits


            loss, aloss, o1loss, o2loss = bc_loss(program, logits)
            print(loss)
            #pdb.set_trace()
            # if epoch % helper.args.print_freq:
            #     lcs_action, lcs_o1, lcs_o2, lcs_triple = utils.computeLCS(program, instructions)
            #     print('Loss {:.3f}. ActionLoss {:.3f}. O1Loss {:.3f}. O2Loss {:.3f}.'
            #           'LCSaction {:.2f}. LCSo1 {:.2f}. LCSo2 {:.2f}. LCStriplet {:.2f}'.format(
            #         loss.data, aloss.data, o1loss.data, o2loss.data,
            #         lcs_action, lcs_o1, lcs_o2, lcs_triple))
            # agent.reset()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test(dataset, helper, policy_net)
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
    m_loss_action = (loss_action * mask).sum(1)/mask.sum(1)
    m_loss_o1 = (loss_o1 * mask).sum(1) / mask.sum(1)
    m_loss_o2 = (loss_o2 * mask).sum(1) / mask.sum(1)
    total_loss = m_loss_action + m_loss_o1 + m_loss_o2
    return total_loss, loss_action, loss_o1, loss_o2

def start():
    helper = utils.setup()
    dataset = EnvDataset(helper.args.dataset_file)
    train(dataset, helper)

    pdb.set_trace()

if __name__ == '__main__':
    start()