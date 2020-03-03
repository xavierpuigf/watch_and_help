import glob
import shutil
import os
file_names = [
     '../../../SyntheticVideos/Output/logs_agent_3_put_fridge/0',
    # '/Users/xavierpuig/Desktop/MultiAgentBench/SyntheticVideos/Output/logs_agent_2_put_fridge/0',
]
def generate_video(image_syn, output_folder, vid_folder, frame_rate):
    import os
    import subprocess
    # curr_folder = os.path.dirname(os.path.realpath(__file__))
    # vid_folder = '{}/../{}/{}/'.format(curr_folder, output_folder, prefix)
    for vid_mod in image_syn:
        subprocess.call(['ffmpeg',
                         '-framerate', str(frame_rate),
                         '-i',
                         '{}/Action_%04d_{}.png'.format(vid_folder, vid_mod),
                         '-pix_fmt', 'yuv420p',
                         '{}/Action_{}.mp4'.format(output_folder, vid_mod)])
        # files_delete = glob.glob('{}/Action_*_{}.png'.format(out_folder, vid_mod))
        # for ft in files_delete: os.remove(ft)
for ij, file_name in enumerate(file_names):
    paths = '{}/*.png'.format(file_name)
    print(paths)
    files = sorted(glob.glob(paths))
    num_files = int(files[-1].split('_')[-2])
    for i in range(1, num_files):
        image_name = '{}/Action_{:04d}_normal.png'.format(file_name, i)
        if not os.path.isfile(image_name):
            print(prev_name, image_name)
            shutil.copy(prev_name, image_name)
            # pdb.set_trace()
        else:
            prev_name = '{}/Action_{:04d}_normal.png'.format(file_name, i)
    fn = file_name.split('/')[-2]
    os.mkdir(fn)
    generate_video(['normal'], file_name.split('/')[-2], file_name, 10)