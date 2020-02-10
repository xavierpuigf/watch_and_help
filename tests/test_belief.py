from vh_graph.envs import belief
import utils_viz
import pdb
import json

if __name__ == '__main__':
    graph_init = 'dataset_toy3/init_envs/TrimmedTestScene1_graph_0.json'
    with open(graph_init, 'r') as f:
        graph = json.load(f)['init_graph']
    bel = belief.Belief(graph)
    new_graph = bel.sample_from_belief()
    pdb.set_trace()
    utils_viz.print_belief(bel)
    utils_viz.print_graph(new_graph)
