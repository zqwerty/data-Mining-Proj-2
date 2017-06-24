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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.decomposition import IncrementalPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import cPickle as pickle
import numpy as np
import time
import sys
import xgboost as xgb
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

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


def classify(classifier, category, testSetNo, foldN, path='../data_tfidf', incremental=True):
    """

    :param classifier: the sklearn classifier object
    :param category: category name to be classified
    :param testSetNo: No. of test set
    :param foldN: # of folds
    :param path: path of labels and matrixxs
    :param incremental: whether to use incremental interface
    :return: precision, recall, f1-measure and used time(in seconds)
    For non-incremental classifiers, directly using original Tf-Idf can result in memory BOOM, so you should use reduced datasets
    For incremental classifiers, no such problem but need some time :)
    """



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

    st = time.clock()

    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF) and (i != testSetNo):
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            data = mat['data']
            if incremental:
                # print "fit %d" % (i)
                classifier.partial_fit(data, yList[i], classes=[0, 1])
                tt = time.clock()
            else:
                mats.append(data)
                yLists.append(yList[i])
            mat.close()

    if not incremental:
        # print 'fit'
        datas = np.vstack(mats)
        labels = np.hstack(yLists)
        st = time.clock()
        classifier.fit(datas, labels)
        tt = time.clock()

    # predict
    testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
    testMat = testMat['data']
    ans = classifier.predict(testMat)

    # calculate correctness
    print classification_report(yList[testSetNo], classifier.predict(testMat))
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



    return precision, recall, f1, tt-st


def onevsrest(classifier, categories, testSetNo, foldN, path='../data_tfidf', incremental=True):


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
            mat = np.zeros([len(labelList), len(categories)], int)
            for j in range(len(labelList)):
                for category in labelList[j][1]:
                    if category in categories:
                        mat[j,categories.index(category)] = 1
                    # mat[j] = 1
            yList[i] = mat
            # training
    mats = []
    yLists = []

    st = time.clock()

    for i in range(foldN):
        if (('vec%d.npz' % (i)) in vecF) and (i != testSetNo):
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            data = mat['data']
            if incremental:
                # print "fit %d" % (i)
                classifier.partial_fit(data, yList[i], classes=[0, 1])
                tt = time.clock()
            else:
                mats.append(data)
                yLists.append(yList[i])
            mat.close()

    if not incremental:
        # print 'fit'
        datas = np.vstack(mats)
        labels = np.concatenate(yLists)
        st = time.clock()
        classifier.fit(datas, labels)
        tt = time.clock()
    # print labels.shape
    # predict
    testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
    testMat = testMat['data']

    # calculate correctness
    print classification_report(yList[testSetNo], classifier.predict(testMat))


    return tt-st


def logistic_regression(categories):
    print 'Logistic Regression'
    classifier = OneVsRestClassifier(sklearn.linear_model.LogisticRegression())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def naive_bayes(categories, origin=False):
    if origin:
        print 'MultinomialNB Naive Bayes'
        for category in categories:
            print category
            classifier = MultinomialNB()
            ans = classify(classifier, category, 2, 10)
            print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    else:
        print 'GaussianNB Naive Bayes'
        classifier = OneVsRestClassifier(GaussianNB())
        ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
        print 'time: %f' % ans


def svm(categories):
    print 'SVM(SVC)'
    classifier = OneVsRestClassifier(sklearn.svm.SVC())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def decision_tree(categories):
    print 'Decision Tree'
    classifier = OneVsRestClassifier(DecisionTreeClassifier())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def mlp(categories, origin=False):
    if origin:
        print 'MLP'
        classifier = OneVsRestClassifier(MLPClassifier())
        ans = onevsrest(classifier, categories, 2, 10)
        print 'time: %f' % ans
    else:
        print 'MLP'
        classifier = OneVsRestClassifier(MLPClassifier())
        ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
        print 'time: %f' % ans


def bootstrap(categories):
    print 'Ensemble bootstrap (Basic estimator: Decision Tree)'
    classifier = OneVsRestClassifier(BaggingClassifier())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def adaboost(categories):
    print 'Adaboost (Basic estimator: Decision Tree)'
    classifier = OneVsRestClassifier(AdaBoostClassifier())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def random_forest(categories):
    print 'Random Forest'
    classifier = OneVsRestClassifier(RandomForestClassifier())
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans


def gradient_boost(categories, testSetNo=2, foldN=10, path='../data_tfidf_reduce', incremental=False):
    # TODO: xgboost
    for category in categories:
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
                    a = 1
                    # print "fit %d" % (i)
                    # classifier.partial_fit(data, yList[i], classes=[0, 1])
                else:
                    mats.append(data)
                    yLists.append(yList[i])
                mat.close()

        if not incremental:
            # print 'fit'
            datas = np.vstack(mats)
            labels = np.hstack(yLists)
            # classifier.fit(datas, labels)
            dtrain = xgb.DMatrix(datas, label=labels)
            param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
            num_round = 2
            bst = xgb.train(param, dtrain, num_round)


        # predict
        testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
        testMat = testMat['data']
        # ans = classifier.predict(testMat)
        dtest = xgb.DMatrix(testMat, label=yList[testSetNo])
        ans = bst.predict(dtest)
        ans[ans < 0.5] = 0
        ans[ans >= 0.5] = 1
        # print sum(ans),ans.shape

        # calculate correctness
        # print classification_report(yList[testSetNo], classifier.predict(testMat))
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
            f1 = 2.0 / (1.0 / precision + 1.0 / recall)

        tt = time.clock()
        ans = precision, recall, f1, tt - st
        print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans


def cluster(cluster_method, categories, foldN=10, path='../data_tfidf_reduce'):


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
    print categories
    for i in range(foldN):
        if ('label%d.p' % (i)) in labelF:
            labelList = pickle.load(open(os.path.join(path, ('label%d.p' % (i)))))
            mat = np.zeros([len(labelList)], int)
            for j in range(len(labelList)):
                # if j > 10:break
                # print labelList[j][1]
                random.shuffle(labelList[j][1])
                for category in labelList[j][1]:
                    if category in categories:
                        mat[j] = categories.index(category)
                        # break
                    else:
                        mat[j] = len(categories)
                # print mat[j]
            yList[i] = mat
        # break
    # training
    mats = []
    yLists = []

    for i in range(foldN):
        if 'vec%d.npz' % (i) in vecF:
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            data = mat['data']
            mats.append(data)
            yLists.append(yList[i])
            mat.close()


    datas = np.vstack(mats)
    labels = np.hstack(yLists)

    st = time.clock()

    res = cluster_method.fit_predict(datas)

    tt = time.clock()

    ami = metrics.adjusted_mutual_info_score(labels, res)
    nmi = metrics.normalized_mutual_info_score(labels, res)

    return res, ami, nmi, tt-st, datas, labels


def kmeans(n_clusters, categories, visual=False):
    if visual:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        ami, nmi, t = visualize(kmeans, categories)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        res, ami, nmi, t, datas, labels = cluster(kmeans, categories)
    print ami
    print nmi
    print t


def visualize(cluster_method, categories, foldN=10, path='../data_tfidf_reduce2'):
    res, ami, nmi, t, datas, labels = cluster(cluster_method, categories, foldN=foldN, path=path)

    reduced_data = datas

    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = cluster_method.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = cluster_method.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    for i in range(len(categories)):
        plt.text(centroids[i, 0], centroids[i, 1], categories[i])
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return ami, nmi, t


def dbscan(n_clusters, categories, visual=False):
    if visual:
        db = DBSCAN(eps=0.3, min_samples=10)
        ami, nmi, t = visualize(db, categories)
    else:
        db = DBSCAN(eps=0.3, min_samples=10)
        res, ami, nmi, t, datas, labels = cluster(db, categories)
    print ami
    print nmi
    print t


if __name__ == '__main__':
    # f = open('log.txt', 'wb')
    # sys.stdout = f
    # preprocess.preprocess()
    #
    categories = pickle.load(open('../data_category/category.p'))
    print categories
    #
    # time = dimensionReduce(2, 10, 2, savePath='../data_tfidf_reduce2')
    # print 'Reduce to 100 dimension time: %f' % (time)

    # kmeans(len(categories)+1, categories)
    # categories = categories[:2]
    # logistic_regression(categories)
    # naive_bayes(categories)
    # svm(categories)
    # decision_tree(categories)
    # mlp(categories)
    # bootstrap(categories)
    # adaboost(categories)
    # random_forest(categories)
    # gradient_boost(categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    cls = len(categories)+1
    # dbscan(cls,categories)
    kmeans(cls,categories)
    # f.close()



