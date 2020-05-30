import json
import pickle as pkl
import pdb
import sys
import argparse
import shutil
import random
import os
from tqdm import tqdm
sys.path.append('../../../virtualhome/simulation')
from unity_simulator import comm_unity
import cv2
import glob
import numpy as np

def write_video(log_file, out_file, comm, file_folder, file_folder_log):
    if 'json' in log_file:
        with open(log_file, 'r') as f:
            content = json.load(f)
    else:
        with open(log_file, 'rb') as f:
            content = pkl.load(f)
    env_id = content['env_id']
    actions = content['action']
    #pdb.set_trace()
    first_obs = content['obs'][0]


    graph = content['init_unity_graph']
    livingroom_center = \
    livingroom_center = \
    [node['bounding_box']['center'] for node in graph['nodes'] if node['class_name'] == 'livingroom'][0]
    if env_id == 5:
        return None
    comm.reset(env_id)
    s, env_graph = comm.environment_graph()
    max_id = sorted([node['id'] for node in env_graph['nodes']])[-1]

    action_1 = content['action'][0]
    action_2 = content['action'][1]

    character_pos1 = [x for x, y in
                      zip([node['bounding_box']['center'] for node in first_obs if node['id'] == 1][0],
                          livingroom_center)]
    # pdb.set_trace()
    character_pos2 = None
    # pdb.set_trace()
    # pdb.set_trace()
    if len(action_2) > 0:
        # pdb.set_trace()
        #character_pos2 = character_pos1
        character_pos2 = [x for x, y in
                          zip([node['bounding_box']['center'] for node in content['obs'][1] if node['id'] == 2][0],
                              livingroom_center)]
        character_pos2[1] = 0.

    # graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in [1, 2]]
    # graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in [1, 2] and edge['to_id'] not in [1, 2]]

    # # shift node
    # for node in graph['nodes']:
    #     if node['id'] > max_id:
    #         node['id'] = node['id'] - max_id + 1000
    # for edge in graph['edges']:
    #     if edge['from_id'] > max_id:
    #         edge['from_id'] = edge['from_id'] - max_id + 1000
    #     if edge['to_id'] > max_id:
    #         edge['to_id'] = edge['to_id'] - max_id + 1000
    print([node['class_name'] for node in graph['nodes'] if 'ceilingfan' in node['class_name']])
    # pdb.set_trace()
    # fan_ids = [node['id'] for node in graph['nodes'] if 'ceilingfan' in node['class_name']]
    # graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in fan_ids]
    # graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in fan_ids and edge['to_id'] not in fan_ids]
    comm.expand_scene(graph)
    # pdb.set_trace()

    cam_id = 81

    cam1 = str(cam_id) # "81"  # "83"
    character_pos1[1] = 0.

    s, im = comm.camera_image([cam1], image_height=960, image_width=1280)
    s, im2 = comm.camera_image([cam1], mode='seg_class', image_height=960, image_width=1280)
    s, im3 = comm.camera_image([cam1], mode='depth', image_height=960, image_width=1280)
    s, im4 = comm.camera_image([cam1], mode='illumination', image_height=960, image_width=1280)
    s, info_cam = comm.camera_data([cam_id])
    cv2.imwrite('{}/{}/top_view.png'.format(file_folder, out_file), im[0])
    cv2.imwrite('{}/{}/top_view_seg.png'.format(file_folder, out_file), im2[0])
    cv2.imwrite('{}/{}/top_view_illum.png'.format(file_folder, out_file), im4[0])
    np.save('{}/{}/top_view_depth.npy'.format(file_folder, out_file), im3[0])

    pdb.set_trace()
    # pdb.set_trace()
    sc = comm.add_character('Chars/Female1', position=character_pos1)
    print("Adding char in pos", character_pos1, env_id)
    if not sc:
        print("Failed to add character")
    # pdb.set_trace()


    if character_pos2 is not None:
        comm.add_character(position=character_pos2)

    s, g = comm.environment_graph()
    # pdb.set_trace()
    s, color_map = comm.instance_colors()
    # pdb.set_trace()
    actions = []
    graphs = [g]
    for it, action in enumerate(tqdm(action_1)):
        # positions.append((action, character_pos1))
        if action is None:
            action_str = ''
        else:
            action_str = '<char0> {}'.format(action).replace('walk', 'walktowards')
        if len(action_2) > 0 and action_2[it] is not None:
            if len(action_str) > 0:
                action_str += ' | '
            action_str += '<char1> {}'.format(action_2[it]).replace('walk', 'walktowards')
        # print('Rendeer...')
        # print(action_str)

        smooth_walk = True

        # continue

        s, m = comm.render_script([action_str],
                           recording=True, gen_vid=False, camera_mode=["PERSON_FROM_BACK"],
                           image_synthesis=['normal'], frame_rate=5, output_folder=file_folder,
                           image_width=512, image_height=384, save_pose_data=True,
                           smooth_walk=smooth_walk, file_name_prefix=out_file, processing_time_limit=250, time_scale=1.)

        # s, m = comm.render_script([action_str],
        #                           recording=True, gen_vid=False, camera_mode=[cam1, cam2], frame_rate=1,
        #                           smooth_walk=False, file_name_prefix=out_file, processing_time_limit=50, time_scale=1.)

        if not s:
            print(m)
            return None
        # pdb.set_trace()
        s, graph = comm.environment_graph()
        graphs.append(graph)
        actions.append(action_str)
        # character_pos1 = [node['bounding_box']['center'] for node in graph['nodes'] if node['id'] == 1][0]

    dict_info = {
        'actions': actions,
        'graphs': graphs,
        'goal': content['gt_goals'],
        'task_name': content['task_name'],
        'color_map': color_map,
        'smooth_walk': smooth_walk,
        'camera_data': info_cam,

    }
    if not os.path.isdir('{}/{}'.format(file_folder_log, out_file)):
        os.makedirs('{}/{}'.format(file_folder_log, out_file))
    print('{}/{}/script_info.json'.format(file_folder_log, out_file))
    with open('{}/{}/script_info.json'.format(file_folder_log, out_file), 'w+') as f:
        f.write(json.dumps(dict_info, indent=4))
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID_ALICE')
    parser.add_argument('--use-editor', action='store_true', default=False, help='whether to use an editor or executable')
    parser.add_argument('--use-docker', action='store_true', default=False, help='whether to use an editor or executable')
    parser.add_argument('--port', default="8180", type=str, help='whether to use an editor or executable')
    parser.add_argument('--display', default="1", type=str, help='what display to use')

    home_path = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge'
    args = parser.parse_args()

    problem_name = 'multiBob_env_task_set_20_check_neurips_test_recursive'

    if args.use_editor:
        comm = comm_unity.UnityCommunication(timeout_wait=250)
    else:
        if args.use_docker:
            comm = comm_unity.UnityCommunication(port=args.port,
                                                 timeout_wait=250, docker_enabled=True)
        else:
            comm = comm_unity.UnityCommunication(x_display=args.display, port=args.port,
                                                file_name='../../../executables/exec_linux.04.27.x86_64', timeout_wait=50)

    if args.use_editor:
        out_folder = '/Users/xavierpuig/Desktop/test_videos/'
        file_names = [
            # '../../record_scratch/rec_good_test/multiAlice_env_task_set_20_check_neurips_test/logs_agent_82_setup_table_0.pik',
            # '../../record_scratch/rec_good_test/multiBob_env_task_set_20_check_neurips_test_recursive/logs_agent_49_prepare_food_0.pik'
            '../../record_scratch/rec_good_test/multiBob_env_task_set_20_check_neurips_test_recursive/logs_agent_1_read_book_0.pik'
            # '../../record_scratch/rec_good_test/multiBob_env_task_set_20_randomgoal/logs_agent_1_read_book_0.pik',
            # '../../record_scratch/rec_good_test/multiBob_env_task_set_20_randomgoal/logs_agent_35_put_dishwasher_0.pik'
        ]
        log_folder = None
    else:
        if args.use_docker:
            out_folder = './unity_vol/videos/'
            log_folder = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/docker_info_script/'
        else:
            out_folder = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/challenge/data_challenge/videos_qualitative/'
            log_folder = out_folder
        file_name_files = home_path + '/split/all_watch.txt'


        files_candidates = glob.glob('../../record_scratch/rec_good_test/{}/logs*.pik'.format(problem_name))
        map_ind_to_file = {(int(file_name.split('_')[2]), int(file_name.split('_')[-1].split('.')[0])): file_name for file_name in files_candidates}
        with open('../../record_scratch/rec_good_test/{}/results_redo.pkl'.format(problem_name), 'rb') as f:
            ct = pkl.load(f)


    for itf, file_name in enumerate(tqdm(file_names)):

        splitf = file_name.split('/')[-1].split('.')[0]
        file_video_gen = '{}/vids_gen/{}.json'.format(home_path, splitf)

        if os.path.isfile(file_video_gen):
            continue
        else:
            if not args.use_editor:
                with open(file_video_gen, 'w+') as f:
                    f.write(json.dumps({splitf: 0}))
            else:
                pass
                #splitf = ['bob_test_5'][itf]

        correct = False
        if args.use_editor:
            correct = write_video(file_name, splitf, comm, out_folder, log_folder)
        else:
            try:
                print("Try ", splitf)
                correct = write_video(file_name, splitf, comm, out_folder, log_folder)
            except:

                print("Failed")
                shutil.rmtree('{}/{}'.format(out_folder, splitf), ignore_errors=True)
                shutil.rmtree('{}/{}'.format(log_folder, splitf), ignore_errors=True)
                if args.use_editor:
                    comm = comm_unity.UnityCommunication()
                else:
                    if args.use_docker:
                        pdb.set_trace()
                        comm = comm_unity.UnityCommunication(port=args.port,
                                                             timeout_wait=250, docker_enabled=True)
                    else:

                        comm.close()
                        comm = comm_unity.UnityCommunication(x_display=args.display, port=args.port,
                                                             file_name='../../../executables/exec_linux.04.27.x86_64',
                                                             timeout_wait=200)

                os.remove(file_video_gen)
                continue

        if correct and not args.use_editor:
            with open(file_video_gen, 'w+') as f:
                f.write(json.dumps({splitf: 1}))