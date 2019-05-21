"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/15 13:28
"""
import numpy as np
import random,h5py
from sklearn.model_selection import train_test_split
from sklearn import neighbors
# from sklearn.metrics import precision_score
import model.svm_classfication as svm_c


def knn_train(feature_all,label_all):
    train_data, _, train_label, _ = train_test_split(feature_all,label_all)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_data,train_label)
    # print(predict)
    # accuracy = precision_score(predict,test_label)
    # print(accuracy)
    # svm_c.plot_confusion_matrix(test_label,predict,['gun','footstep','vehicle','other'],True)
    # load test data
    feature, label = svm_c.read_h5py('feature_test.h5')
    print(feature.shape, label.shape)
    feature_all = []
    label_all = []
    count = [0, 0, 0, 0]
    for i in range(len(label)):
        if label[i] == 0:
            count[0] += 1
            if count[0] > 100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 1:
            count[1] += 1
            if count[1] > 100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 2:
            count[2] += 1
            if count[2] > 100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 9:
            count[3] += 1
            if count[3] > 100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
    # print(np.shape(feature_all))
    print(len(label_all))
    id = np.arange(0, len(label_all))
    random.shuffle(id)
    in_feature = []
    in_label = []
    for i in range(len(id)):
        in_feature.append(feature_all[id[i]])
        in_label.append(label_all[id[i]])
    predict = knn.predict(in_feature)
    svm_c.plot_confusion_matrix(in_label, predict, ['gun', 'footstep', 'vehicle', 'other'], True)

    return predict


def read_h5py(filename):
    with h5py.File(filename, 'r') as f:
        # def prtname(name):
        #     print(name)
        # f.visit(prtname)
        subgroup = f['subgroup']
        data1 = subgroup['feature']
        data2 = subgroup['label']
        feature = data1.value
        label = data2.value
        # print("data1 feature:", feature)
        print("data1 label:", label)
    return feature,label


def avg_frame(data):
    avg_data = []
    for i in range(data.shape[1]):
        avg_data.append(np.average(data[:, i]))
    return avg_data


if __name__ == '__main__':
    feature,label = read_h5py('feature.h5')
    print(feature.shape,label.shape)
    feature_all = []
    label_all = []
    for i in range(len(label)):
        if label[i]==0:
            feature_all.append(avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i]==1:
            feature_all.append(avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i]==2:
            feature_all.append(avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i]==9:
            feature_all.append(avg_frame(feature[i]))
            label_all.append(label[i])
    # print(np.shape(feature_all))
    print(len(label_all))
    id = np.arange(0,len(label_all))
    random.shuffle(id)
    in_feature = []
    in_label = []
    for i in range(len(id)):
        in_feature.append(feature_all[id[i]])
        in_label.append(label_all[id[i]])

    predict = knn_train(in_feature,in_label)



