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
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, AgglomerativeClustering, Birch
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
        f1 = 2.0 / (1.0/precision + 1.0/recall)

    return precision, recall, f1, tt-st


def onevsrest(classifiers, categories, testSetNo, foldN, path='../data_tfidf', incremental=True):


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
                # classifier.partial_fit(data, yList[i], classes=[0, 1])
                # TODO:incremental
                tt = time.clock()
            else:
                mats.append(data)
                yLists.append(yList[i])
            mat.close()

    sc = StandardScaler()
    train_time = []
    if not incremental:
        # print 'fit'
        datas = np.vstack(mats)
        labels = np.concatenate(yLists)
        sc = sc.fit(datas)
        datas = sc.transform(datas)

        for classifier in classifiers:
            st = time.clock()
            classifier.fit(datas, labels)
            tt = time.clock()
            train_time.append(tt-st)
    # print labels.shape
    # predict
    testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
    testMat = testMat['data']
    testMat = sc.transform(testMat)
    # calculate correctness
    for classifier in classifiers:
        print classification_report(yList[testSetNo], classifier.predict(testMat))#, target_names=categories)
        print 'train time: %f' % train_time[classifiers.index(classifier)]
    return train_time


def logistic_regression(categories):
    print 'Logistic Regression'
    classifier = OneVsRestClassifier(sklearn.linear_model.LogisticRegression())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def naive_bayes(categories, origin=False):
    if origin:
        print 'MultinomialNB Naive Bayes'
        for category in categories:
            # print category
            classifier = MultinomialNB()
            ans = classify(classifier, category, 2, 10)
            print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
    else:
        print 'GaussianNB Naive Bayes'
        classifier = OneVsRestClassifier(GaussianNB())
        ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
        print 'time: %f' % ans[0]


def svm(categories):
    print 'SVM(SVC)'
    classifier = OneVsRestClassifier(sklearn.svm.SVC(kernel='linear'))
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def decision_tree(categories):
    print 'Decision Tree'
    classifier = OneVsRestClassifier(DecisionTreeClassifier())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def mlp(categories, origin=False):
    if origin:
        print 'MLP'
        for category in categories:
            # print category
            classifier = MLPClassifier()
            ans = classify(classifier, category, 2, 10)
            print 'Precision: %f Recall: %f F1-measure: %f time: %f' % ans
        # classifier = OneVsRestClassifier(MLPClassifier())
        # ans = onevsrest([classifier], categories, 2, 10)
        # print 'time: %f' % ans[0]
    else:
        print 'MLP'
        classifier = OneVsRestClassifier(MLPClassifier(max_iter=1000))
        ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
        print 'time: %f' % ans[0]


def knn(categories):
    print 'kNN'
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def linear_discriminant_analysis(categories):
    print 'LinearDiscriminantAnalysis'
    classifier = OneVsRestClassifier(LinearDiscriminantAnalysis())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]



def test(categories):
    classifier = [OneVsRestClassifier(sklearn.linear_model.LogisticRegression()), OneVsRestClassifier(DecisionTreeClassifier())]
    ans = onevsrest(classifier, categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    # for a in ans:
    # print 'time: %f' % ans[0]


def bootstrap(categories):
    print 'Ensemble bootstrap (Basic estimator: Decision Tree)'
    classifier = OneVsRestClassifier(BaggingClassifier())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def adaboost(categories):
    print 'Adaboost (Basic estimator: Decision Tree)'
    classifier = OneVsRestClassifier(AdaBoostClassifier())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def random_forest(categories):
    print 'Random Forest'
    classifier = OneVsRestClassifier(RandomForestClassifier())
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def gradient_boost_sklearn(categories):
    print 'gradient boost sklearn'
    classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0))
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def bagging(categories):
    print 'bagging: kNN'
    classifier = OneVsRestClassifier(BaggingClassifier(KNeighborsClassifier(5), max_features=10, n_jobs=-1))
    ans = onevsrest([classifier], categories, 2, 10, path='../data_tfidf_reduce', incremental=False)
    print 'time: %f' % ans[0]


def gradient_boost(categories, testSetNo=2, foldN=10, path='../data_tfidf_reduce'):
    # TODO: xgboost
    print 'xgboost'
    st = time.clock()
    s = [0,0,0]
    for category in categories:


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
                mats.append(data)
                yLists.append(yList[i])
                mat.close()

        # print 'fit'
        sc = StandardScaler()
        datas = np.vstack(mats)
        sc.fit(datas)
        datas = sc.transform(datas)
        labels = np.hstack(yLists)
        # classifier.fit(datas, labels)
        dtrain = xgb.DMatrix(datas, label=labels)
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        num_round = 20
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        bst = xgb.train(param, dtrain, num_round)


        # predict
        testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
        testMat = testMat['data']
        testMat = sc.transform(testMat)
        # ans = classifier.predict(testMat)
        dtest = xgb.DMatrix(testMat, label=yList[testSetNo])
        ans = bst.predict(dtest)
        # print ans
        # print sum(ans), ans.shape
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

        ans = precision, recall, f1
        for i in range(3):
            s[i] += ans[i]
        print 'Precision: %.2f Recall: %.2f F1-measure: %.2f' % ans
    tt = time.clock()
    s = map(lambda x:x/len(categories),s)
    print 'avg / total: Precision: %.2f Recall: %.2f F1-measure: %.2f' % (s[0],s[1],s[2])
    print 'train time: %f' % (tt-st)


def cluster(cluster_method, categories, foldN=10, path='../data_tfidf_reduce', visual=False):


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

    if visual:
        datas = datas[:, :2]
        datas = MinMaxScaler().fit_transform(datas)

    st = time.clock()

    res = cluster_method.fit_predict(datas)

    tt = time.clock()

    ami = metrics.adjusted_mutual_info_score(labels, res)
    nmi = metrics.normalized_mutual_info_score(labels, res)

    return res, ami, nmi, tt-st, datas, labels


def visualize(cluster_method, categories, foldN=10, path='../data_tfidf_reduce'):
    res, ami, nmi, t, datas, labels = cluster(cluster_method, categories, foldN=foldN, path=path, visual=True)
    print datas
    reduced_data = datas

    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    # if hasattr(cluster_method, 'labels_'):
    #     Z = cluster_method.labels_.astype(np.int)
    # else:
    #     Z = cluster_method.predict(np.c_[xx.ravel(), yy.ravel()])
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

    # for i in range(len(categories)):
    #     plt.text(centroids[i, 0], centroids[i, 1], categories[i])
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()

    return ami, nmi, t


def kmeans(n_clusters, categories, visual=False, init='random'):
    if visual:
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42)
        ami, nmi, t = visualize(kmeans, categories)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42)
        res, ami, nmi, t, datas, labels = cluster(kmeans, categories,path='../data_tfidf_reduce')
    print ami
    print nmi
    print t


def dbscan(n_clusters, categories, visual=False):
    if visual:
        db = DBSCAN(eps=0.3, min_samples=10)
        ami, nmi, t = visualize(db, categories)
    else:
        db = DBSCAN(eps=0.05, min_samples=3)
        res, ami, nmi, t, datas, labels = cluster(db, categories)
    print ami
    print nmi
    print t


def affinity_propagation(n_clusters, categories, visual=False):
    if visual:
        cls_method = AffinityPropagation(damping=.9,preference=-200)
        ami, nmi, t = visualize(cls_method, categories)
    else:
        cls_method = Birch(n_clusters=n_clusters, threshold=0.01)
        res, ami, nmi, t, datas, labels = cluster(cls_method, categories)
    print ami
    print nmi
    print t


def spectral_clustering(n_clusters, categories, visual=False):
    if visual:
        cls_method = SpectralClustering(n_clusters=n_clusters, n_jobs=-1)
        ami, nmi, t = visualize(cls_method, categories)
    else:
        cls_method = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1)
        res, ami, nmi, t, datas, labels = cluster(cls_method, categories)
    print ami
    print nmi
    print t


def classify_visualize(classifier, category):
    foldN = 10
    path = '../data_tfidf_reduce'
    testSetNo = 2
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
        if ('vec%d.npz' % (i)) in vecF:
            mat = np.load(open(os.path.join(path, ('vec%d.npz' % (i))), 'rb'))
            data = mat['data']
            mats.append(data)
            yLists.append(yList[i])
            mat.close()

    # print 'fit'
    datas = np.vstack(mats)
    labels = np.hstack(yLists)

    datas = datas[:, :2]
    datas = MinMaxScaler().fit_transform(datas)


    st = time.clock()
    classifier.fit(datas, labels)
    tt = time.clock()

    # predict
    # testMat = np.load(open(os.path.join(path, ('vec%d.npz' % (testSetNo))), 'rb'))
    # testMat = testMat['data']
    # ans = classifier.predict(testMat)

    X = datas
    y = labels
    print classification_report(labels, classifier.predict(datas))


    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"

    plt.figure(figsize=(10, 5))

    # Plot the decision boundaries
    plt.subplot(121)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="Class %s" % n)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    # Plot the two-class decision scores
    twoclass_output = classifier.decision_function(X)
    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.subplot(122)
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[y == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='Class %s' % n,
                 alpha=.5)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.show()


if __name__ == '__main__':
    # f = open('log.txt', 'wb')
    # sys.stdout = f
    # preprocess.preprocess()
    #
    categories = pickle.load(open('../data_category/category.p'))
    cls = len(categories) + 1
    print categories
    #
    # time = dimensionReduce(2, 10, 2, savePath='../data_tfidf_reduce2')
    # print 'Reduce to 100 dimension time: %f' % (time)

    # #-t0
    # logistic_regression(categories)
    # naive_bayes(categories)
    # svm(categories)
    # decision_tree(categories)
    # mlp(categories)
    # knn(categories)
    # linear_discriminant_analysis(categories)
    # bootstrap(categories)
    # adaboost(categories)
    # random_forest(categories)
    # gradient_boost(categories, 2, 10, path='../data_tfidf_reduce')
    # gradient_boost_sklearn(categories)
    # bagging(categories)

    # naive_bayes(categories, origin=True)
    # mlp(categories, origin=True)

    # cluster
    # dbscan(cls, categories)
    # kmeans(cls,categories)

    # visualize classifier
    # classify_visualize(sklearn.linear_model.LogisticRegression(), categories[17])
    # classify_visualize(LinearDiscriminantAnalysis(), categories[17])
    # classify_visualize(AdaBoostClassifier(),categories[17])
    # classify_visualize(GradientBoostingClassifier(), categories[17])

    # visualize cluster
    # kmeans(cls,categories,visual=True)

    # f.close()



