import os
import numpy as np
import cv2

class Dataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'Insert a valid path!'

        self.data_classes = os.listdir(self.data_path)

        self.data_mapping = {}

        for c, c_name in enumerate(self.data_classes):
            # Jump invisible files
            if c_name[0] == '.':
                continue

            temp_path = os.path.join(self.data_path, c_name)

            temp_images = os.listdir(temp_path)

            for i in temp_images:
                img_tmp = os.path.join(temp_path, i)

                if img_tmp.endswith('.jpg'):
                    if c_name == 'distractor':
                        self.data_mapping[img_tmp] = -1
                    else:
                        self.data_mapping[img_tmp] = int(c_name)

        print('Loaded {:d} images from {:s} '.format(len(self.data_mapping.keys()),
                                                    self.data_path))

    def get_data_paths(self):
        images = []
        classes = []
        for img_path in self.data_mapping.keys():
            if img_path.endswith('.jpg'):
                images.append(img_path)
                classes.append(self.data_mapping[img_path])
        return images, np.array(classes)

    def num_classes(self):
        return len(self.data_classes)

class BaseSolution(object):
    def __init__(self, training_dataset, query_dataset, gallery_dataset):
        self.training_paths, self.training_classes = training_dataset.get_data_paths()
        self.query_paths, self.query_classes = query_dataset.get_data_paths()
        self.gallery_paths, self.gallery_classes = gallery_dataset.get_data_paths()

        self.args = None
        self.parse_args()
    
    def solve(self):
        '''
        Should return gallery_matches
        '''
        pass

    def print_result(self):
        '''
        Print top k accuracy
        '''
        pass

    def parse_args(self):
        '''
        Parse user argument
        '''
        pass

def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp