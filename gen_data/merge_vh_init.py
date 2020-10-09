import pickle as pkl
import glob

files = glob.glob('../../data_challenge/init_envs/5*.p')
content_all = []
for file_name in files:
    with open(file_name, 'rb') as f:
        content = pkl.load(f)
    print(file_name, len(content))
    content_all += content

print(len(content_all))
with open('../../data_challenge/init_envs/5_envs_total.p', 'wb') as f:
    pkl.dump(content_all, f)