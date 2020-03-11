def inside_not_trans(graph):
    id2node = {node['id']: node for node in graph['nodes']}
    parents = {}
    grabbed_objs = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':

            if edge['from_id'] not in parents:
                parents[edge['from_id']] = [edge['to_id']]
            else:
                parents[edge['from_id']] += [edge['to_id']]
        elif edge['relation_type'].startswith('HOLDS'):
            grabbed_objs.append(edge['to_id'])

    edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
            if len(parents[edge['from_id']]) == 1:
                edges.append(edge)
            else:
                if edge['from_id'] > 1000:
                    pdb.set_trace()
        else:
            edges.append(edge)
    graph['edges'] = edges

    # # add missed edges
    # missed_edges = []
    # for obj_id, action in self.obj2action.items():
    #     elements = action.split(' ')
    #     if elements[0] == '[putback]':
    #         surface_id = int(elements[-1][1:-1])
    #         found = False
    #         for edge in edges:
    #             if edge['relation_type'] == 'ON' and edge['from_id'] == obj_id and edge['to_id'] == surface_id:
    #                 found = True
    #                 break
    #         if not found:
    #             missed_edges.append({'from_id': obj_id, 'relation_type': 'ON', 'to_id': surface_id})
    # graph['edges'] += missed_edges

    parent_for_node = {}

    char_close = {1: [], 2: []}
    for char_id in range(1, 3):
        for edge in graph['edges']:
            if edge['relation_type'] == 'CLOSE':
                if edge['from_id'] == char_id and edge['to_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['to_id'])
                elif edge['to_id'] == char_id and edge['from_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['from_id'])
    ## Check that each node has at most one parent
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
                print('{} has > 1 parent'.format(edge['from_id']))
                pdb.set_trace()
                raise Exception
            parent_for_node[edge['from_id']] = edge['to_id']
            # add close edge between objects in a container and the character
            if id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave',
                                                        'dishwasher', 'stove']:
                for char_id in range(1, 3):
                    if edge['to_id'] in char_close[char_id] and edge['from_id'] not in char_close[char_id]:
                        graph['edges'].append({
                            'from_id': edge['from_id'],
                            'relation_type': 'CLOSE',
                            'to_id': char_id
                        })
                        graph['edges'].append({
                            'from_id': char_id,
                            'relation_type': 'CLOSE',
                            'to_id': edge['from_id']
                        })

    ## Check that all nodes except rooms have one parent
    nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
    nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
    nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
    if len(nodes_without_parent) > 0:
        for nd in nodes_without_parent:
            print(id2node[nd])
        pdb.set_trace()
        raise Exception

    return graph