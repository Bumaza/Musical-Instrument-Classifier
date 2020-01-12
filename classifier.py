#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A Simple Musical Instrument Classifier

"""

__author__ = "Jozef Budac"

import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix


"""Musical instruments contained in the dataset"""
class_names = ['violin', 'viola', 'cello', 'double-bass', 'guitar', 'clarinet', 'saxophone']


def detect_leading_silence(sound, silence_treshold=.001):
    """
    To avoid noise or silence detects leading intensity of the sound
    which is less as some constant value(silence_treshold).
    """
    trim, sound = 0, np.array(sound/max(sound))
    while sound[trim] < silence_treshold:
        trim += 1

    return trim


def feature_extraction():
    """Takes Mel Frequency Spectrum as features"""

    if not os.path.exists('./single_tone_data.csv'):
       return np.loadtxt('single_tone_data.csv', delimiter=';')

    instruments = {'violin': 1, 'viola': 2, 'cello': 3, 'double-bass': 4,
                   'guitar': 5, 'clarinet': 6, 'saxophone': 7}

    instruments_counts = [0] * 7

    files = glob.glob('./data/data/*/*.mp3')
    np.random.shuffle(files)

    data = []

    num_of_ex = 0

    for filename in files:
        try:
            music, sr = librosa.load(filename, sr=None)

            start_trim = detect_leading_silence(music)
            end_trim = detect_leading_silence(np.flipud(music))

            duration = len(music)
            y = music[start_trim:duration-end_trim]

            mfcc = librosa.feature.mfcc(y=y, sr=sr) #, n_mfcc=13
            mfcc = np.mean(mfcc, axis=1)

            feature = mfcc.reshape(20)

            label = 0
            for instrument, value in instruments.items():
                if instrument in filename:
                    label = value
                    instruments_counts[value - 1] += 1
                    break

            data.append([filename, feature, label])

        except Exception as ex:
            num_of_ex += 1
            continue

    with open('single_tone_data.csv', 'w') as file:
        for d in data:
            file.write(';'.join(str(f) for f in d[1]) + ';' + str(d[2]) + '\n')

    plt.title('Number of examples')
    plt.ylabel('Count')
    plt.bar(class_names, instruments_counts,
            color=('teal', 'aquamarine', 'coral', 'gold', 'crimson', 'violet', 'wheat'))
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(0, max(instruments_counts), max(instruments_counts) // 10))
    plt.show()

    return feature_extraction()


def unpack(data):
    """Split data to features and labels (X and y)"""
    return data[..., :-1], data[..., -1]


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{0:.2f}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def main():
    """
    Comparison of three kind of models SVM, k-NN and Random Forest.
    As result, get the accuracy rate of each model.
    """
    data = np.array(feature_extraction())

    np.random.shuffle(data)

    s_60p, s_20p = int(len(data) * 0.6), int(len(data) * 0.2)

    train_data = data[:s_60p]  # 60%
    validation_data = data[s_60p : s_60p+s_20p]  # 20%
    test_data = data[s_60p+s_20p:] # 20%

    X_train, y_train = unpack(train_data)
    X_validation, y_validation = unpack(validation_data)
    X_test, y_test = unpack(test_data)

    svc = svm.SVC() #
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(n_estimators=20, max_depth=50, warm_start=True)

    svc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    svm_predicted = svc.predict(X_validation)
    knn_predicted = knn.predict(X_validation)
    rf_predicted = rf.predict(X_validation)

    svm_accuracy = np.sum(svm_predicted == y_validation) / len(y_validation) * 100
    knn_accuracy = np.sum(knn_predicted == y_validation) / len(y_validation) * 100
    rf_accuracy = np.sum(rf_predicted == y_validation) / len(y_validation) * 100

    svm_pcm = plot_confusion_matrix(svc, X_test, y_test, display_labels=class_names, xticks_rotation='vertical', normalize='true')
    knn_pcm = plot_confusion_matrix(knn, X_test, y_test, display_labels=class_names, xticks_rotation='vertical', normalize='true')
    rf_pcm = plot_confusion_matrix(rf, X_test, y_test, display_labels=class_names, xticks_rotation='vertical', normalize='true')

    svm_pcm.ax_.set_title('Normalized confusion matrix for SVM')
    knn_pcm.ax_.set_title('Normalized confusion matrix for KNN')
    rf_pcm.ax_.set_title('Normalized confusion matrix for Random Forest')

    plt.show()

    model_names = ['SVM', 'Random Forest', 'KNN']
    model_accuracy = [svm_accuracy, rf_accuracy, knn_accuracy]

    plt.title('Accuracy rate')
    plt.ylabel('Prediction [%]')
    rects = plt.bar(model_names, model_accuracy, color=('teal', 'coral', 'khaki'))
    autolabel(rects)
    plt.xticks(np.arange(len(model_names)), model_names)
    plt.yticks(np.arange(0, 100, 10))
    plt.show()

    print('SVM accuracy rate: {0:.2f}%'.format(svm_accuracy ))
    print('RandomForest accuracy rate: {0:.2f}%'.format(knn_accuracy))
    print('KNN accuracy rate: {0:.2f}%'.format(rf_accuracy))


if __name__ == '__main__':
    main()