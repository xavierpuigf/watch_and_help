import json
import pdb
from environment import Environment
from single_agent import SingleAgent


class DataPoint():
    def __init__(self, env, program):
        self.env = env
        self.program = program

def read_problem(file_problem):
    # This should go in a dataset class
    with open(file_problem, 'r') as f:
        problems = json.load(f)

    problems_dataset = []
    for problem in problems:
        goal_file = problem['file_name']
        graph_file = problem['env_path']
        goal_name = problem['goal']
        program_file = problem['program']

        with open(goal_file, 'r') as f:
            goal_str = f.read()

        with open(program_file, 'r') as f:
            program = f.readlines()


        problems_dataset.append(
            {
                'goal': goal_str,
                'graph_file': graph_file,
                'goal_name': goal_name,
                'program': program
            }
        )
    return problems_dataset


def train(datapoints):
    # Think how to handle this with multiple envs. One agent multiple envs?

    num_rollouts = 5
    num_epochs = 10
    for epoch in range(num_epochs):
        for dp in datapoints:
            curr_env = dp.env
            agent = curr_env.agents[0]
            for it in range(num_rollouts):
                # Decide object
                observations = agent.get_observations()
                instr_info = agent.get_instruction(observations)

                instr = instr_info['instruction']
                r, states, infos = curr_env.env.step(instr)
                agent.policy_net.update_info(instr_info, r)

            loss = agent.policy_net.bc_loss(dp.program)
        pdb.set_trace()



def test():
    problems = read_problem('dataset/example_problem.json')
    dps = []
    for problem in problems:
        graph_file, goal, program = problem['graph_file'], problem['goal'], problem['program']
        env = Environment(graph_file, goal, 2)
        dp = DataPoint(env, program)
        dps.append(dp)
    train(dps)

    pdb.set_trace()

if __name__ == '__main__':
    test()