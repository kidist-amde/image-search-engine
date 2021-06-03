import torch
import torchvision
import torch.utils.data as data
from glob import glob
import cv2
from torchvision.transforms import ToTensor
import os
class Dataset(data.Dataset):
    '''
    There is a problem, the images' numbers and  sizes of different classes are differentiated
    '''

    def __init__(self, data_root):
        data_root_list = glob(data_root+'/*')
        data_root_list.sort()
        self.img_list = []
        self.label_list = []
        # self.names = {0: 'airplane', 1: 'bird', 2: 'car', 3: 'cat', 4: 'city', 5: 'deer', 6: 'dog', 7: 'face',
        #               8: 'green', 9: 'horse',
        #               10: 'house_building', 11: 'house_indoor', 12: 'monkey', 13: 'office', 14: 'sea', 15: 'ship',
        #               16: 'truck'}
        self.num_classes = len(data_root_list)
        for i, filename in enumerate(data_root_list):
            basename = os.path.basename(filename)
            temp_img_list = glob(filename+'/*.*')
            temp_img_list.sort()
            temp_label_list = [i]*len(temp_img_list)
            self.img_list += temp_img_list
            self.label_list += temp_label_list
        assert len(self.img_list) == len(self.label_list)
        print('all image lens: {}'.format(len(self.img_list)))
        self.totensor = ToTensor()
        self.__getitem__(1)
        self.__getitem__(2)

    def __getitem__(self, item):
        img = cv2.imread(self.img_list[item])
        img = cv2.resize(img, (100, 100))
        label = self.label_list[item]
        #print('label:', label)
        tensor = self.totensor(img)
        return {'tensor': tensor, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_list)

if __name__ == '__main__':
    data_root = 'code/train'
    dataset = Dataset(data_root)
    pass

