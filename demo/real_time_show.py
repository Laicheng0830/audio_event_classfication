"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/15 14:35
"""

import matplotlib.pyplot as plt
import numpy as np
import time,random
from math import *
from feature_extract import mel_feature
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import model.svm_classfication as svm_c


def data_norm(data):
    # data in array
    data = data/2**15
    return data

def show_figure():
    plt.ion() #start interactive mode
    plt.figure(1)
    t = [0]
    m = []

    for i in range(2000):
        plt.clf()
        t_now = i*0.1
        t.append(t_now)
        m.append(sin(t))
        plt.plot(t,m,'-r')
        plt.draw()
        time.sleep(0.01)

def function(x):
    return x - 1

def predict_svm(test_data,test_tag):
    clf = joblib.load("30ms_svm_model.pkl")
    start = time.time()
    print(time.localtime(time.time()))
    print(time.asctime(time.localtime(time.time())))
    predict = clf.predict(test_data)
    end = time.time()
    print(time.localtime(time.time()))
    print(time.asctime(time.localtime(time.time())))
    print("frame run times ms :",((end-start)/len(test_data))*1000)
    print("run times s",(end-start),len(test_data))

    ac_score = metrics.accuracy_score(test_tag,predict)
    cl_report = metrics.classification_report(test_tag,predict)
    print(ac_score)
    print(cl_report)
    print(confusion_matrix(test_tag,predict))
    print("succeed")
    return predict

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_pred, y_true)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

if __name__ == '__main__':
    feature, label = svm_c.read_h5py('feature_test.h5')
    print(feature.shape, label.shape)
    feature_all = []
    label_all = []
    count = [0,0,0,0]
    for i in range(len(label)):
        if label[i] == 0:
            count[0]+=1
            if count[0]>100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 1:
            count[1]+=1
            if count[1]>100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 2:
            count[2]+=1
            if count[2]>100:
                continue
            feature_all.append(svm_c.avg_frame(feature[i]))
            label_all.append(label[i])
        if label[i] == 9:
            count[3]+=1
            if count[3]>100:
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

    # svm_train(in_feature,in_label)
    predict = predict_svm(test_data=in_feature, test_tag=in_label)
    # print(predict)
    # print(in_label)
    plot_confusion_matrix(predict,in_label,['gun','footstep','vehicle','other'],True)
