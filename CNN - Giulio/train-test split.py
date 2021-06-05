import os
import glob
import shutil
import math

full = glob.glob(os.path.join('./dataset', 'train', '*'))
test = os.path.join('./dataset', 'test')

for cla in full:
    file_list = glob.glob(os.path.join(cla, '*'))
    test_dir = os.path.join(test, (os.path.basename(cla)))

    test_size = math.floor(math.log2(len(file_list)))
    os.makedirs(test_dir)

    for i in range(test_size):
        file = file_list[i]
        shutil.move(file, test_dir)


