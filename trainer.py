import json
import pdb
import torch
import utils
import argparse
from envdataset import EnvDataset
from environment import Environment
from models.single_policy import SinglePolicy



def train(dataset, helper):
    # Think how to handle this with multiple envs. One agent multiple envs?
    # Here we set the policy and the env
    policy_net = SinglePolicy(dataset, helper)
    envs = []
    for dp in dataset:
        graph_file, goal, program = dp
        curr_env = Environment(graph_file, goal, 2)
        for agent in curr_env.agents:
            agent.policy_net = policy_net
            agent.activation_info = policy_net.activation_info()
        envs.append(curr_env)

    optimizer = torch.optim.Adam(list(policy_net.parameters()))

    num_rollouts = helper.args.num_rollouts
    num_epochs = helper.args.num_epochs
    for epoch in range(num_epochs):
        for it, dp in enumerate(dataset):
            graph_file, goal, program = dp
            curr_env = envs[it]
            agent = curr_env.agents[0]
            for it in range(num_rollouts):
                # Decide object
                observations = agent.get_observations()
                instr_info = agent.get_instruction(observations)

                instr = instr_info['instruction']
                r, states, infos = curr_env.env.step(instr)
                agent.update_info(instr_info, r)

            loss = agent.policy_net.bc_loss(program, agent.agent_info)
            print(loss)
            agent.reset()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




def test():
    helper = utils.setup()
    dataset = EnvDataset(helper.args.dataset_file)
    train(dataset, helper)

    pdb.set_trace()

if __name__ == '__main__':
    test()