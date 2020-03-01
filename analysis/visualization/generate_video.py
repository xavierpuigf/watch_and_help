import json
import pickle
import pdb
import sys

sys.path.append('../../../virtualhome/simulation')

from unity_simulator import comm_unity

def write_video(log_file, out_file, comm):
    with open(log_file, 'r') as f:
        content = json.load(f)

    print(out_file)
    env_id = content['env_id']
    actions = content['action']
    graph = content['init_unity_graph']
    livingroom_center = \
    [node['bounding_box']['center'] for node in graph['nodes'] if node['class_name'] == 'livingroom'][0]
    comm.reset(env_id)
    s, env_graph = comm.environment_graph()
    max_id = sorted([node['id'] for node in env_graph['nodes']])[-1]

    action_1 = content['action']['0']
    action_2 = content['action']['1']

    character_pos1 = [x + y for x, y in
                      zip([node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 1][0],
                          livingroom_center)]
    character_pos2 = None
    if len(action_1) > 0:
        # pdb.set_trace()
        character_pos2 = character_pos1
        character_pos2 = [x + y for x, y in
                          zip([node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 2][0],
                              livingroom_center)]
        character_pos2[1] = 0.

    graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in [1, 2]]
    graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in [1, 2] and edge['to_id'] not in [1, 2]]

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
    comm.add_character('Chars/Female1', position=character_pos1)



    if character_pos2 is not None:
        comm.add_character(position=character_pos2)

    positions = []

    for it, action in enumerate(action_1):
        positions.append((action, character_pos1))
        action_str = '<char0> {}'.format(action).replace('walk', 'walktowards')
        if len(action_2) > 0:
            action_str += '| <char1> {}'.format(action_2[it]).replace('walk', 'walktowards')
        print('Rendeer...')
        comm.render_script([action_str],
                           recording=True, gen_vid=False, camera_mode="PERSON_TOP",
                           smooth_walk=True, file_name_prefix=out_file, processing_time_limit=50)
        s, graph = comm.environment_graph()
        character_pos1 = [node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 1][0]

    # with open('trace.json', 'w+') as f:
    #     f.write(json.dumps(positions, indent=4))


if __name__ == '__main__':
    file_name = '../logs_bob/logs_agent_2_read_book.json'

    #comm = comm_unity.UnityCommunication(x_display="3", port="8079",
    #                                     file_name='../../../executables/exec_linux03.1/exec_linux03.1.x86_64')

    comm = comm_unity.UnityCommunication(port="8090")
    splitf = file_name.split('/')[-1].split('.')[0]
    write_video(file_name, splitf, comm)


