"""
zqwerty & llylly
Data Mining Proj 2
Data Classfier Module
"""

'''
    sklearn dependency
'''
import os
import sklearn
from sklearn.naive_bayes import GaussianNB
import cPickle as pickle
import numpy as np
import time

import preprocess

def classify(classifier, category, testSetNo, foldN = 10, path = '../data_tfidf'):
    """

    :param classifier: the sklearn classifer object
    :param category: category name to be classified
    :param testSetNo: No. of test set
    :param foldN: # of folds
    :param path: path of labels and matrixs
    :return: precision, recall, f1-measure and used time(in seconds)
    """

    st = time.clock()

    # label files
    labelF = []
    for root, dirs, files in os.walk(path):
        labelF = [x for x in files if 'p' in x]

    # matrix files
    vecF = []
    for root, dirs, files in os.walk(path):
        vecF = [x for x in files if 'npz' in x]

    # generate label(0 or 1) vector from labels in files
    yList = {}
    for i in range(foldN):
        if ('label%d.p' % (i)) in labelF:
            labelList = pickle.load(open(os.path.join(path, ('label%d.p' % (i)))))
            mat = np.zeros([len(labelList)], int)
            for j in range(len(labelList)):
                if category in labelList[j][1]:
                    mat[j] = 1
            yList[i] = mat

    # training
    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF) and (i != testSetNo):
            print "fit %d" % (i)
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            mat = mat['data']
            classifier.fit(mat, yList[i])

    # predict
    testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
    testMat = testMat['data']
    ans = classifier.predict(testMat)

    # calculate correctness
    truePosi = ans * yList[testSetNo]
    falsePosi = ans * (1 - yList[testSetNo])
    falseNega = (1 - ans) * yList[testSetNo]

    precision = 0
    if (sum(truePosi) + sum(falsePosi)) > 0:
        precision = (float)(sum(truePosi)) / (float)(sum(truePosi) + sum(falsePosi))
    recall = 0
    if (sum(truePosi) + sum(falseNega)) > 0:
        recall = (float)(sum(truePosi)) / (float)(sum(truePosi) + sum(falseNega))
    f1 = 1.0 / (1.0/precision + 1.0/recall)

    tt = time.clock()

    return precision, recall, f1, tt-st

if __name__ == '__main__':
    preprocess.preprocess()


    categories = pickle.load(open('../data_category/category.p'))
    print categories

    '''
    print 'Logistic Regression'
    classifier = sklearn.linear_model.LogisticRegression()
    ans = classify(classifier, 'paid death notices', 2, 10)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''
    '''
    print 'Gaussian Naive Bayes'
    classifier = GaussianNB()
    ans = classify(classifier, 'paid death notices', 2, 10)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    print 'SVM(SVC)'
    classifier = sklearn.svm.SVC()
    ans = classify(classifier, 'paid death notices', 2, 10)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans

