import sys
sys.path.append('..')
from base import BaseSolution
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from scipy import spatial

class Histogram:
    def __init__(self, bins):
        self.bins = bins

    def detectAndCompute(self, image, other):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape).flatten()
        hist = hist[np.newaxis, :]
        return None, hist

class FeatureExtractor(object):

    def __init__(self, feature_extractor, gray=False):

        self.feature_extractor = feature_extractor
        self.gray = gray

    def get_descriptor(self, img_path):
        img = cv2.imread(img_path)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs

    def extract_features(self, data_list):
        # we init features
        features = []

        for i, img_path in enumerate(tqdm(data_list, desc='Extraction')):
            # get descriptor
            histo = self.get_descriptor(img_path)
            features.append(histo)
        features = np.concatenate(features)

        return features

class HistogramSolution(BaseSolution):

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Challenge presentation example')
        parser.add_argument('--data_path',
                            '-d',
                            type=str,
                            default='challenge_data_small',
                            help='Dataset path')
        parser.add_argument('--descriptor',
                            '-desc',
                            type=str,
                            default='sift',
                            help='Descriptor to be used')
        parser.add_argument('--output_dim',
                            '-o',
                            type=int,
                            default=10,
                            help='Descriptor length')
        parser.add_argument('--save_dir',
                            '-s',
                            type=str,
                            default=None,
                            help='Save or not gallery/query feats')
        parser.add_argument('--gray',
                            '-g',
                            action='store_true',
                            help='Grayscale/RGB SIFT')
        self.args = parser.parse_args()

    def solve(self):
        feature_extractor = Histogram(bins=[32, 32, 32])

        # we define the feature extractor providing the model
        extractor = FeatureExtractor(feature_extractor=feature_extractor, gray=self.args.gray)

        # now we can use features
        # we get query features
        query_features = extractor.extract_features(self.query_paths)

        # we get gallery features
        gallery_features = extractor.extract_features(self.gallery_paths)

        print(gallery_features.shape, query_features.shape)

        pairwise_dist = spatial.distance.cdist(query_features, gallery_features, 'minkowski', p=2.)

        print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))
        indices = np.argsort(pairwise_dist, axis=-1)

        gallery_matches = self.gallery_classes[indices]
        return gallery_matches