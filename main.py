import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np



def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)




def project():
    np.random.seed(42)

    first_name = 'Michal'
    last_name = 'Pogodski'

    data_path = Path('data')
    data_path = os.getenv('DATA_PATH', data_path)
    x, y = load_dataset(data_path)

    feature_detector_descriptor = cv2.AKAZE_create()

    with Path('vocab_model.p').open('rb') as vocab_file:
        vocab_model = pickle.load(vocab_file)
    x_transformed = apply_feature_transform(x, feature_detector_descriptor, vocab_model)

    with Path('clf.p').open('rb') as classifier_file:
        clf = pickle.load(classifier_file)

    score = clf.score(x_transformed, y)

    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:
        json.dump({'score': score}, score_file)


if __name__ == '__main__':
    project()