import sys
sys.path.append('..')
from base import BaseSolution
import imagehash
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

def extract_hash(gallery_paths, query_paths, hashfunc):
    '''
    If we use perceptive hash algorithm, then we don't need training set
    '''

    gallery_features = []
    # calculate each image in gallery's hash value
    for image_path in gallery_paths:
        hash_result = hashfunc(Image.open(image_path)).hash.flatten()
        gallery_features.append([hash_result])
    gallery_features = np.concatenate(gallery_features)

    query_features = []
    # then query features, hard to call feature, because only single value
    for image_path in query_paths:
        hash_result = hashfunc(Image.open(image_path)).hash.flatten()
        query_features.append([hash_result])
    query_features = np.concatenate(query_features)

    # we can set different distance calcualte method here
    pairwise_dist = cdist(query_features, gallery_features, 'hamming')
    print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))
    indices = np.argsort(pairwise_dist, axis=-1)
    
    # indices = np.random.randint(len(self.gallery_paths),
    #                             size=(len(self.query_paths), len(self.gallery_paths)))
    return indices
    

class AhashSolution(BaseSolution):
    def solve(self):
        indices = extract_hash(self.gallery_paths, self.query_paths, hashfunc=imagehash.average_hash)
        gallery_matches = self.gallery_classes[indices]
        return gallery_matches

class PhashSolution(BaseSolution):
    def solve(self):
        indices = extract_hash(self.gallery_paths, self.query_paths, hashfunc=imagehash.phash)
        gallery_matches = self.gallery_classes[indices]
        return gallery_matches

class DhashSolution(BaseSolution):
    def solve(self):
        indices = extract_hash(self.gallery_paths, self.query_paths, hashfunc=imagehash.dhash)
        gallery_matches = self.gallery_classes[indices]
        return gallery_matches