import os
import glob
import shutil
import json

exp_name = 'logs_agent_env3_milk_milkshake_wineglass'

folder_name = '/Users/xavierpuig/Desktop/MultiAgentBench/SyntheticVideos/Output/file_{}/0'.format(exp_name) 
print(folder_name)
frames = sorted(glob.glob('{}/*.png'.format(folder_name)))

print(len(frames))
print(frames[-1])
max_num = int(frames[-1].split('/')[-1].split('_')[1].split('_')[0])
rt = frames[0].split('Action')[0] 
for frame_num in range(max_num):
	prev_file_name = '{}/Action_{:04}_normal.png'.format(rt, frame_num-1)
	file_name = '{}/Action_{:04}_normal.png'.format(rt, frame_num)
	if not os.path.isfile(file_name):
		print(file_name)
		shutil.copy(prev_file_name, file_name)


with open('{}/ftaa_file_{}.txt'.format(rt, exp_name), 'r') as f:
	lines = f.readlines()
	prog = [x.split()[1:] for x in lines if x.split()[0] != '-1']

with open('../../data/{}.json'.format(exp_name), 'r') as f:
	content = json.load(f)

prog = [p for p in prog if int(p[1]) not in [25, 106, 138, 188]]
action = content['action']['0']
plan = content['plan']['0']

for i in range(len(prog)):
	frames = int(prog[i][1]), int(prog[i][2])
	plan_c = plan[i]
	act = action[i]
	print(frames, act)
import pdb
pdb.set_trace()
