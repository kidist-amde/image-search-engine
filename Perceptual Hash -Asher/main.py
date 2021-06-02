import os
import numpy as np
from base import Dataset, topk_accuracy
from methods import (
    KmeansSolution,
    HistogramSolution,
    AhashSolution,
    PhashSolution,
    DhashSolution
)

TOP_K = [1, 3, 10]
METHODS_TO_RUN = {
    # 'kmeans': {
    #     'class_name': KmeansSolution,
    #     'args': ''
    # },
    # 'histogram': {
    #     'class_name': HistogramSolution,
    #     'args': ''
    # },
    'ahash': {
        'class_name': AhashSolution,
        'args': ''
    },
    'phash': {
        'class_name': PhashSolution,
        'args': ''
    },
    'dhash': {
        'class_name': DhashSolution,
        'args': ''
    }
}
MATCH_RESULT = {}

def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'ex1/dataset')
    # we define training dataset
    training_path = os.path.join(data_path, 'training')
    # we define validation dataset
    validation_path = os.path.join(data_path, 'validation')
    query_path = os.path.join(validation_path, 'query')
    gallery_path = os.path.join(validation_path, 'gallery')

    training_dataset = Dataset(data_path=training_path)
    query_dataset = Dataset(data_path=query_path)
    gallery_dataset = Dataset(data_path=gallery_path)

    return training_dataset, query_dataset, gallery_dataset

def main():
    
    training_dataset, query_dataset, gallery_dataset = load_dataset()
    query_paths, query_classes = query_dataset.get_data_paths()

    for method_name in METHODS_TO_RUN:
        MATCH_RESULT[method_name] = METHODS_TO_RUN[method_name]['class_name'](training_dataset, query_dataset, gallery_dataset).solve()
    
    print('########## RESULTS ##########')
    for method_name, gallery_matches in MATCH_RESULT.items():
        print(f'METHOD: {method_name}')

        for k in TOP_K:
            topk_acc = topk_accuracy(query_classes, gallery_matches, k)
            print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))
        print()


if __name__ == '__main__':
    main()
