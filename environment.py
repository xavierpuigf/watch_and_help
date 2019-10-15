from single_agent import SingleAgent
import gym
import vh_graph

class Environment():
    def __init__(self, graph_file, goal, num_agents):
        self.env = gym.make('vh_graph-v0')
        print(goal)
        print(graph_file)
        self.env.reset(graph_file, goal)
        self.env.to_pomdp()
        self.agents = []
        for it in range(num_agents):
            self.agents.append(SingleAgent(self.env, goal, it))

