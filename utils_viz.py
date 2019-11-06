import graphviz
import json
import pdb
import random

def getclass(node):
    return '{}\n{}'.format(node['class_name'], node['id'])

def delete_redundant_edges_and_ids(graph):
    class_nodes_delete = ['wall', 'floor', 'ceiling', 'door', 'curtain', 'window', 'doorjamb']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete]

    graph['nodes'] = [x for x in graph['nodes'] if x['id'] not in ids_delete]
    graph['edges'] = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    parent_node = {}
    children_node = {}
    for it, edge in enumerate(graph['edges']):
        if edge['relation_type'] == 'INSIDE':
            if edge['to_id'] not in parent_node.keys(): parent_node[edge['to_id']] = []
            parent_node[edge['to_id']].append(edge['from_id'])
            if edge['from_id'] not in children_node.keys(): children_node[edge['from_id']] = []
            children_node[edge['from_id']].append(edge['to_id'])

    final_edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            all_parents = children_node[edge['from_id']]
            all_children = parent_node[edge['to_id']]
            if len(set(all_parents).intersection(all_children)) > 0:
                continue
        # else:
        #     if ((edge['from_id'] in children_node and edge['to_id'] in children_node[edge['from_id']]) or
        #             (edge['to_id'] in children_node and edge['from_id'] in children_node[edge['to_id']])):
        #         continue

        final_edges.append(edge)
    graph['edges'] = final_edges
    return graph

def graph2im(graph, id_char=None, id_goal=None):
    """
    Outputs an image given a graph
    :param graph:
    :return:
    """

    # graph = delete_redundant_edges_and_ids(graph)

    class_nodes_delete = ['wall', 'floor', 'ceiling', 'door', 'curtain']
    ids_delete = [x['id'] for x in graph['nodes'] if x['class_name'] in class_nodes_delete]

    graph['nodes'] = [x for x in graph['nodes'] if x not in ids_delete]
    graph['edges'] = [x for x in graph['edges'] if x['from_id'] not in ids_delete and x['to_id'] not in ids_delete]

    id2node = {x['id']: x for x in graph['nodes']}

    g = graphviz.Digraph(engine='dot')
    g.attr(compound='true')

    # g.attr(rank='min')

    container_nodes = list(set([x['to_id'] for x in graph['edges'] if x['relation_type'] == 'INSIDE']))
    children = {}
    parent = {}
    num_children_subgraph = {}
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['to_id'] not in children:
                children[edge['to_id']] = []
                num_children_subgraph[edge['to_id']] = 0
            parent[edge['from_id']] = edge['to_id']
            children[edge['to_id']].append(edge['from_id'])
            if edge['from_id'] in container_nodes:
                num_children_subgraph[edge['to_id']] += 1

    subgraphs_added = {}
    curr_subgraphs = [x for (x,y) in num_children_subgraph.items() if y == 0]
    # pdb.set_trace()
    while len(curr_subgraphs) > 0:
        next_subgraphs = []
        for curr_subgraph_id in curr_subgraphs:
            name_graph = getclass(id2node[curr_subgraph_id])
            cng = graphviz.Digraph(name='cluster_'+str(curr_subgraph_id))
            cng.attr(label=name_graph)
            cng.node(name=str(curr_subgraph_id), style='invis')
            # pdb.set_trace()
            children_c_graph = children[curr_subgraph_id]
            for child in children_c_graph:
                if child in subgraphs_added:
                    cng.subgraph(subgraphs_added[child])

                else:
                    if id_char is not None and child == id_char:
                        cng.node(name=str(child), label=getclass(id2node[child]), color='darkseagreen', style='filled')

                    else:
                        cng.node(name=str(child), label=getclass(id2node[child]))

            subgraphs_added[curr_subgraph_id] = cng
            if curr_subgraph_id == 1:
                pdb.set_trace()
            if curr_subgraph_id not in parent:
                g.subgraph(cng)
                continue
            parent_graph = parent[curr_subgraph_id]
            num_children_subgraph[parent_graph] -= 1
            if num_children_subgraph[parent_graph] == 0:
                next_subgraphs.append(parent_graph)
        curr_subgraphs = next_subgraphs

    colors = {
        'INSIDE': 'orange',
        'ON': 'blue',
        'CLOSE': 'purple',
        'FACING': 'red',
        'BETWEEN': 'green'

    }
    style = {
        'INSIDE': '',
        'ON': '',
        'CLOSE': 'invis',
        'FACING': 'invis',
        'BETWEEN': 'invis'

    }
    print('Edges...')
    max_num = 2
    for edge in graph['edges']:
        rt = edge['relation_type']
        if rt != 'INSIDE' and edge['from_id'] not in ids_delete and edge['to_id'] not in ids_delete:
            if rt == 'CLOSE':
                # print(edge['from_id'], edge['to_id'], edge['relation_type'])
                if parent[edge['from_id']] != parent[edge['to_id']]:
                    continue
                if random.randint(0,max_num) < max_num:
                    continue
            # if ((edge['to_id'] in parent and parent[edge['to_id']] == edge['from_id']) or
            #    (edge['from_id'] in parent and parent[edge['from_id']] == edge['to_id'])):
            #     continue

            if edge['from_id'] not in children and edge['to_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt], style=style[rt])

            elif edge['from_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       lhead='cluster_' + str(edge['to_id']), style=style[rt])
            elif edge['to_id'] not in children:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       ltail='cluster_' + str(edge['from_id']), style=style[rt])
            else:
                g.edge(str(edge['from_id']), str(edge['to_id']), color=colors[rt],
                       ltail='cluster_'+str(edge['from_id']), lhead='cluster_'+str(edge['to_id']), style=style[rt])
    return g


def print_graph(graph, output='graph_example.gv'):
    id_char = [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
    g = graph2im(graph, id_char)
    g.render(output)

if __name__ == '__main__':
    input_graph = 'dataset_toy3/init_envs/TrimmedTestScene5_graph_8.json'
    with open(input_graph, 'r') as f:
        graph = json.load(f)
    g = graph2im(graph['init_graph'])
    g.render('graph_example.gv')