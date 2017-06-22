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
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import IncrementalPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle
import numpy as np
import time

import preprocess

def dimensionReduce(testSetNo, foldN, dimension, path = '../data_tfidf', savePath = '../data_tfidf_reduce'):
    """

    :param testSetNo: index of test set
    :param foldN: # of folds
    :param dimension: target dimension to be reduced to
    :param path: source tfidf folder
    :param savePath: new tfidf folder
    :return: time in seconds
    Simply reduction the tf-idf matrix to target dimension using incremental PCA, all others are same as data_tfidf folder except the matrix dimension
    """
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    st = time.clock()

    # label files
    labelF = []
    for root, dirs, files in os.walk(path):
        labelF = [x for x in files if 'p' in x]

    # matrix files
    vecF = []
    for root, dirs, files in os.walk(path):
        vecF = [x for x in files if 'npz' in x]

    # simply copy labels
    for i in range(foldN):
        if ('label%d.p' % (i)) in labelF:
            labelList = pickle.load(open(os.path.join(path, ('label%d.p' % (i)))))
            pickle.dump(labelList, open(os.path.join(savePath, ('label%d.p' %(i))), 'wb'))

    PCA = IncrementalPCA(n_components = dimension)

    # training (excluding the test set)
    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF) and (i != testSetNo):
            print "fit %d" % (i)
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i)))))
            data = mat['data']
            PCA.partial_fit(data)
            mat.close()

    # save
    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF):
            print "save %d" % (i)
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i)))))
            res = PCA.transform(mat['data'])
            np.savez(open(os.path.join(savePath, ('vec%d.npz' % (i))), 'wb'), data = res, no = mat['no'])
            mat.close()

    tt = time.clock()
    return tt-st


l:
    """

    :param classifier: the sklearn classifier object
    :param category: category name to be classified
    :param testSetNo: No. of test set
    :param foldN: # of folds
    :param path: path of labels and matrix
    :param incremental: whether to use incremental interface
    :return: precision, recall, f1-measure and used time(in seconds)
    For non-incremental classifiers, directly using original Tf-Idf can result in memory BOOM, so you should use reduced datasets
    For incremental classifiers, no such problem but need some time :)
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
    mats = []
    yLists = []

    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF) and (i != testSetNo):
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            data = mat['data']
            if incremental:
                print "fit %d" % (i)
                classifier.partial_fit(data, yList[i], classes=[0, 1])
            else:
                mats.append(data)
                yLists.append(yList[i])
            mat.close()

    if not incremental:
        print 'fit'
        datas = np.vstack(mats)
        labels = np.hstack(yLists)
        classifier.fit(datas, labels)

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
    f1 = 0
    if (precision > 0) and (recall > 0):
        f1 = 2.0 / (1.0/precision + 1.0/recall)

    tt = time.clock()

    return precision, recall, f1, tt-st


if __name__ == '__main__':
    preprocess.preprocess()


    categories = pickle.load(open('../data_category/category.p'))
    print categories

    '''
    time = dimensionReduce(2, 10, 100)
    print 'Reduce to 100 dimension time: %f' % (time)
    '''

    '''
    print 'Logistic Regression'
    classifier = sklearn.linear_model.LogisticRegression()
    ans = classify(classifier, 'paid death notices', 2, 10, path = '../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'MultinomialNB Naive Bayes'
    classifier = MultinomialNB()
    ans = classify(classifier, 'paid death notices', 2, 10)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'SVM(SVC)'
    classifier = sklearn.svm.SVC()
    ans = classify(classifier, 'paid death notices', 2, 10, path = '../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'Decision Tree'
    classifier = DecisionTreeClassifier()
    ans = classify(classifier, 'paid death notices', 2, 10, path = '../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'MLP'
    classifier = MLPClassifier()
    ans = classify(classifier, 'paid death notices', 2, 10)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'Ensemble bootstrap (Basic estimator: Decision Tree)'
    classifier = BaggingClassifier()
    ans = classify(classifier, 'paid death notices', 2, 10, path = '../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'Adaboost (Basic estimator: Decision Tree)'
    classifier = AdaBoostClassifier()
    ans = classify(classifier, 'paid death notices', 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    '''
    print 'Random Forest'
    classifier = RandomForestClassifier()
    ans = classify(classifier, 'paid death notices', 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    '''

    # TODO: xgboost

    # TODO: clusters
