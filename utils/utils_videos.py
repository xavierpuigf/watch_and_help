import shutil
import sys
import os
import subprocess
import numpy as np
import cv2

def create_video(video_folder, format, out_file):
    file_and_format = f'{video_folder}/{format}'
    frames = 500
    for i in range(1, 500):
        print(file_and_format)
        frame_num = file_and_format.replace('*', '{:04}'.format(i))
        prev_frame_num = file_and_format.replace('*', '{:04}'.format(i-1))
        print(prev_frame_num)
        if not os.path.isfile(frame_num):
            shutil.copy(prev_frame_num, frame_num)

    frame_rate = 10
    subprocess.call(['ffmpeg',
                    '-framerate', str(frame_rate),
                    '-i',
                     '{}/{}'.format(video_folder, format.replace('*', '%04d')),
                     '-pix_fmt', 'yuv420p',
                     '{}.mp4'.format(out_file)])


def merge_frames(in_formats, nframes, out_format):
    for i in range(0, nframes):
        curr_imgs = []
        for in_format in in_formats:

            frame_num = in_format.replace('*', '{:04}'.format(i))
            prev_frame_num = in_format.replace('*', '{:04}'.format(i - 1))

            if not os.path.isfile(frame_num):
                shutil.copy(prev_frame_num, frame_num)
            curr_imgs.append(cv2.imread(frame_num))
        img_join = np.concatenate(curr_imgs, 1)
        cv2.imwrite(out_format.replace('*', '{:04}'.format(i)), img_join)

if __name__ == '__main__':
    # merge_frames(['/Users/xavierpuig/Desktop/test_videos/bob_with_info/action_*.png',
    #               '/Users/xavierpuig/Desktop/test_videos/alice_with_info/action_*.png'], 570,
    #              '/Users/xavierpuig/Desktop/test_videos/merged2_info/Action_*_normal.png')
    # create_video('/Users/xavierpuig/Desktop/test_videos/merged2_info/',
    #              'Action_*_normal.png',
    #              '/Users/xavierpuig/Desktop/test_videos/alice_and_bob_info.mp4')
    create_video('/Users/xavierpuig/Desktop/test_videos/alice_with_info_belief/',
                 'action_*.png',
                 '/Users/xavierpuig/Desktop/test_videos/alice_info_goals.mp4')
    #create_video(sys.argv[1], sys.argv[2], sys.argv[3])