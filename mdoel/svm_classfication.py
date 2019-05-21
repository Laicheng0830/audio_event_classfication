"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/14 10:07
"""

from sklearn import cross_validation, svm, metrics
from sklearn.metrics import confusion_matrix
import time,h5py,random
from sklearn.externals import joblib
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


TIME = 30

# def svm_train(train_data,train_label,test_data,test_label):
def svm_train(feature_all, label_all):
    # data_all = train_data+test_data
    # label_train1 = np.hstack((train_label,test_label))
    # train_data, test_data, train_label, test_label = train_test_split(feature_all,label_all)
    train_data, test_data, train_label, test_label = cross_validation.train_test_split(feature_all, label_all)

    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # for each combination of parameters
            clf_t = svm.SVC(gamma=gamma, C=C,decision_function_shape='ovo')
            clf_t.fit(train_data, train_label)
            score = clf_t.score(test_data, test_label)
            if score > 0.9:
                joblib.dump(clf_t, str(TIME) + str(score) + "ms " + "svm_model.pkl")
                break
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
                print("current score:",score)
    print("find best parameters")
    print("best score: ", best_score)
    print("best parameters: ", best_parameters)

    print("start SVC")
    clf = svm.SVC(gamma = best_parameters['gamma'],C = best_parameters['C'],decision_function_shape='ovo')
    clf.fit(train_data,train_label)
    joblib.dump(clf, str(TIME)+"ms "+"svm_model.pkl")

    predict = clf.predict(test_data)
    # predicted data
    ac_score = metrics.accuracy_score(test_label, predict)
    # Build test accuracy
    cl_report = metrics.classification_report(test_label, predict)
    # Generate cross validation reports
    print(ac_score)
    # Display data accuracy
    print(cl_report)

def predict_svm(test_data,test_tag):
    # clf = svm.SVC(gamma=10, C=0.1, decision_function_shape='ovo')#,class_weight={0:2,1:2})
    # clf.fit(train_data, train_tag)
    clf = joblib.load("30ms_svm_model.pkl")
    start = time.time()
    # print(time.localtime(time.time()))
    # print(time.asctime(time.localtime(time.time())))
    predict = clf.predict(test_data)
    end = time.time()
    # print(time.localtime(time.time()))
    # print(time.asctime(time.localtime(time.time())))
    # print("frame run times ms :",((end-start)/len(test_data))*1000)
    # print("run times s",(end-start),len(test_data))

    ac_score = metrics.accuracy_score(test_tag,predict)
    cl_report = metrics.classification_report(test_tag,predict)
    # print(ac_score)
    # print(cl_report)
    # print(confusion_matrix(test_tag,predict))
    # print("succeed")
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
    cm = confusion_matrix(y_true, y_pred)
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



    svm_train(in_feature,in_label)
    # predict_svm(test_data=in_feature,test_tag=in_label)