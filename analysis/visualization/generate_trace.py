import json
import pickle
import pdb
import sys

sys.path.append('../../../virtualhome/simulation')

from unity_simulator import comm_unity

if __name__ == '__main__':
    file_name = '../../../record/init7_read_book_1_full/logs_agent_2_read_book.json'
    with open(file_name, 'r') as f:
        content = json.load(f)

    comm = comm_unity.UnityCommunication(x_display="4", port="8079", file_name='../../../executables/exec_linux02.29/exec_linux02.29.x86_64')

    env_id = content['env_id']
    actions = content['action']
    graph = content['init_unity_graph']
    livingroom_center = [node['bounding_box']['center'] for node in graph['nodes'] if node['class_name'] == 'livingroom'][0]
    comm.reset(env_id)
    s, env_graph = comm.environment_graph()
    max_id = sorted([node['id'] for node in env_graph['nodes']])[-1]

    character_pos1 = [x+y for x, y in zip([node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 1][0], livingroom_center)]

    graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in [1,2]]
    graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in [1,2] and edge['to_id'] not in [1,2]]

    # shift node
    for node in graph['nodes']:
        if node['id'] > max_id:
            node['id'] = node['id'] - max_id + 1000
    for edge in graph['edges']:
        if edge['from_id'] > max_id:
            edge['from_id'] = edge['from_id'] - max_id + 1000
        if edge['to_id'] > max_id:
            edge['to_id'] = edge['to_id'] - max_id + 1000

    comm.expand_scene(graph)

    character_pos1[1] = 0.
    comm.add_character(position=character_pos)
    cam_id = 78
    s, image = comm.camera_image([cam_id])


positions = []
action = content['action']['0']
for action in content['action']['0']:

    positions.append((action, character_pos1))
    print(character_pos)
    comm.render_script(['<char0> {}'.format(action).replace('walk', 'walktowards')], recording=False, gen_vid=False)
    s, graph = comm.environment_graph()
    character_pos1 = [node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 1][0]

with open('trace.json', 'w+') as f:
    f.write(json.dumps(positions, indent=4))