import glob
import shutil
import os
file_name = 'Output/put_fridge/0'
paths = '{}/*.png'.format(file_name)
print(paths)
files = glob.glob(paths)

num_files = int(files[-1].split('_')[-2])
for i in range(1, num_files):
    image_name = '{}/Action_{:04d}_normal.png'.format(file_name, i)
    prev_name = '{}/Action_{:04d}_normal.png'.format(file_name, i-1)
    if not os.path.isfile(image_name):
        shutil.copy(prev_name, image_name)
