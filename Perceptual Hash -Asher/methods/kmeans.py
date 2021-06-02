import sys
sys.path.append('..')
from base import BaseSolution
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from scipy import spatial
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse

class FeatureExtractor(object):

    def __init__(self, feature_extractor, model, out_dim=20, scale=None,
                 subsample=100, gray=False):

        self.feature_extractor = feature_extractor
        self.model = model
        self.scale = scale
        self.subsample = subsample
        self.gray = gray

    def get_descriptor(self, img_path):
        img = cv2.imread(img_path)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs

    def fit_model(self, data_list):
        training_feats = []
        # we extact SIFT descriptors
        for img_path in tqdm(data_list, desc='Fit extraction'):
            descs = self.get_descriptor(img_path)
            
            if descs is None:
                continue
            
            if self.subsample:
                # TODO: change here
                sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                descs = descs[sub_idx, :]

            training_feats.append(descs)
        training_feats = np.concatenate(training_feats)
        print('--> Model trained on {} features'.format(training_feats.shape))
        # we fit the model
        self.model.fit(training_feats)
        print('--> Model fitted')

    def fit_scaler(self, data_list):
        features = self.extract_features(data_list)
        print('--> Scale trained on {}'.format(features.shape))
        self.scale.fit(features)
        print('--> Scale fitted')

    def extract_features(self, data_list):
        # we init features
        features = np.zeros((len(data_list), self.model.n_clusters))

        for i, img_path in enumerate(tqdm(data_list, desc='Extraction')):
            # get descriptor
            descs = self.get_descriptor(img_path)
            # 2220x128 descs
            preds = self.model.predict(descs)
            histo, _ = np.histogram(preds, bins=np.arange(self.model.n_clusters+1), density=True)
            # append histogram
            features[i, :] = histo

        return features

    def scale_features(self, features):
        # we return the normalized features
        return self.scale.transform(features)

class KmeansSolution(BaseSolution):

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Challenge presentation example')
        parser.add_argument('--data_path',
                            '-d',
                            type=str,
                            default='dataset',
                            help='Dataset path')
        parser.add_argument('--output_dim',
                            '-o',
                            type=int,
                            default=20,
                            help='Descriptor length')
        parser.add_argument('--save_dir',
                            '-s',
                            type=str,
                            default=None,
                            help='Save or not gallery/query feats')
        self.args = parser.parse_args()

    def solve(self):
        feature_extractor = cv2.SIFT_create()

        # we define model for clustering
        model = KMeans(n_clusters=self.args.output_dim, n_init=10, max_iter=5000, verbose=False)
        # model = MiniBatchKMeans(n_clusters=self.args.output_dim, random_state=0, batch_size=100, max_iter=100, verbose=False)
        scale = StandardScaler()

        # we define the feature extractor providing the model
        extractor = FeatureExtractor(feature_extractor=feature_extractor,
                                     model=model,
                                     scale=scale,
                                     out_dim=self.args.output_dim)

        # we fit the KMeans clustering model
        extractor.fit_model(self.training_paths)
        
        extractor.fit_scaler(self.training_paths)
        # now we can use features
        # we get query features
        query_features = extractor.extract_features(self.query_paths)
        query_features = extractor.scale_features(query_features)

        # we get gallery features
        gallery_features = extractor.extract_features(self.gallery_paths)
        gallery_features = extractor.scale_features(gallery_features)

        print(gallery_features.shape, query_features.shape)
        pairwise_dist = spatial.distance.cdist(query_features, gallery_features, 'minkowski', p=2.)
        print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))
        indices = np.argsort(pairwise_dist, axis=-1)

        gallery_matches = self.gallery_classes[indices]
        return gallery_matches