import gym
import vh_graph
import json
import pdb
from models.single_agent import SingleAgent
import torch

def read_problem(file_problem):
    # This should go in a dataset class
    with open(file_problem, 'r') as f:
        problems = json.load(f)

    problems_dataset = []
    for problem in problems:
        goal_file = problem['file_name']
        graph_file = problem['env_path']
        goal_name = problem['goal']

        with open(goal_file, 'r') as f:
            goal_str = f.read()

        problems_dataset.append(
            {
                'goal': goal_str,
                'graph_file': graph_file,
                'goal_name': goal_name
            }
        )
    return problems_dataset


def train(envs):

    # Think how to handle this with multiple envs. One agent multiple envs?
    curr_env = envs[0]
    num_rollouts = 5
    # need to reuse the policy between the agents
    agent = SingleAgent(curr_env, curr_env.task, 0)
    for it in range(num_rollouts):
        observations = agent.get_observations()
        # Decide object
        instr = agent.get_instruction(observations)
        pdb.set_trace()




def test():
    problems = read_problem('dataset/example_problem.json')
    problem = problems[0]
    env = gym.make('vh_graph-v0')
    env.reset(problem['graph_file'], problem['goal'])
    env.to_pomdp()
    train([env])

    pdb.set_trace()

if __name__ == '__main__':
    test()