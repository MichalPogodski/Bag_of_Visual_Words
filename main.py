import json
import os
import pickle
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import svm,  ensemble, cluster
from sklearn.tree import DecisionTreeClassifier
import cv2
import numpy as np


# TODO: versions of libraries that will be used:
#  Python 3.9 (you can use previous versions as well)
#  numpy 1.19.4
#  scikit-learn 0.22.2.post1
#  opencv-python 4.2.0.34


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(dataset_dir_path.iterdir()):
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


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here
    return x


def project():
    np.random.seed(42)

    # TODO: fill the following values
    first_name = 'Michal'
    last_name = 'Pogodski'

    data_path = Path('/home/michal/RiSA/sem2/ZPO/BoVW_project/train')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)


    # TODO: create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.AKAZE_create()

    # TODO: train a vocabulary model and save it using pickle.dump function
    train_images, test_images, train_labels, test_labels = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)


    train_descriptor = []
    for image in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptor.append(descriptor)

    kmeans = cluster.KMeans(n_clusters=128, random_state=42)
    kmeans.fit(train_descriptor)

    x_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
    y_train = train_labels

    classifier = svm.SVC(kernel='poly', random_state=42)
    classifier.fit(x_train, y_train)

    pickle.dump(classifier, open('./clf.p', 'wb'))
    pickle.dump(kmeans, open('vocab_model.p', 'wb'))

    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)
    x_transformed = apply_feature_transform(test_images, feature_detector_descriptor, vocab_model)

####################################################
    # x_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    # y_train = train_labels
    #
    # clf_DT = DecisionTreeClassifier(random_state=42)
    # clf_DT.fit(x_train, y_train)
    #
    # clf_svm = svm.SVC(random_state=42)
    # clf_svm.fit(x_train, y_train)
    #
    # clf_rf = ensemble.RandomForestClassifier(random_state=42)
    # clf_rf.fit(x_train, y_train)


    # print('DT: ', clf_DT.score(x_transformed, test_labels))
    # print('SVC: ', clf_svm.score(x_transformed, test_labels))
    # print('rf: ', clf_rf.score(x_transformed, test_labels))

#####################################################
####################################################
    # x_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    # y_train = train_labels
    #
    # clf_0 = svm.SVC(kernel='poly', random_state=42)
    # clf_0.fit(x_train, y_train)
    #
    # clf_1 = svm.SVC(kernel='linear', random_state=42)
    # clf_1.fit(x_train, y_train)
    #
    # clf_2 = svm.SVC(kernel='rbf', random_state=42)
    # clf_2.fit(x_train, y_train)
    #
    # clf_3 = svm.SVC(kernel='sigmoid', random_state=42)
    # clf_3.fit(x_train, y_train)
    #
    # clf_4 = svm.SVC(kernel='precomputed', random_state=42)
    # clf_4.fit(x_train, y_train)
    #
    # print('clf_0: ', clf_0.score(x_transformed, test_labels))
    # print('clf_1: ', clf_1.score(x_transformed, test_labels))
    # print('clf_2: ', clf_2.score(x_transformed, test_labels))
    # print('clf_3: ', clf_3.score(x_transformed, test_labels))
    # print('clf_4: ', clf_4.score(x_transformed, test_labels))

#####################################################



    # TODO: train a classifier and save it using pickle.dump function
    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(x_transformed, test_labels) ###############################################
    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:  # Don't change the path here
        json.dump({'score': score}, score_file)


if __name__ == '__main__':
    project()