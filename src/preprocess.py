"""
zqwerty & llylly
Data Mining Proj 2
Data Preprocessing Module
"""

import os
import re
import cPickle as pickle
import unicodedata
import random
import time
import string

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


'''
NOTICE: enchant package is required for spelling check
'''
import enchant
'''
NOTICE: NLTK package data is required to run this script
'''
import nltk
import math
import numpy as np


def test():
    tree = ET.ElementTree(file='../data_slice/1801293.xml')
    root = tree.getroot()
    head = root.find('head')
    body = root.find('body')
    title = head.find('title').text
    doc_id = head.find('docdata').find('doc-id').attrib['id-string']
    locations = []
    for c in tree.iter('location'):
        locations.append(c.text)
    for c in head.findall('meta'):
        if 'publication_day_of_month' == c.attrib['name']:
            print c.attrib['content']
        if 'publication_month' == c.attrib['name']:
            print c.attrib['content']
        if 'publication_year' == c.attrib['name']:
            print c.attrib['content']
        if 'online_sections' == c.attrib['name']:
            print c.attrib['content']
    full_text = ""
    for c in tree.iter('block'):
        if 'full_text' == c.attrib['class']:
            for c2 in c.findall('p'):
                full_text += c2.text
    print full_text


def readSingleXML(absoPath):
    """
    :param absoPath: the absolute path of the XML to read
    :return: (docid, category, full_text)
    """
    tree = ET.parse(absoPath)
    # doc-id
    for c in tree.iter('doc-id'):
        docid = c.attrib['id-string']

    # category -list
    category = []
    for c in tree.find('head').findall('meta'):
        if 'online_sections' == c.attrib['name']:
            oric = str(c.attrib['content'])
            oric = oric.replace('(', ', ')
            oric = oric.replace(')', ', ')
            oric = re.split(';|and|,', oric)
            for cc in oric:
                cc = cc.strip()
                if len(cc) > 0:
                    category.append(cc.lower())

    # full-text
    full_text = ''
    for c in tree.iter('block'):
        if 'full_text' == c.attrib['class']:
            for c2 in c.findall('p'):
                # replace non-ascii characters
                s = c2.text
                if type(s) == unicode:
                    s = unicodedata.normalize('NFKD', c2.text).encode('ascii', 'ignore')
                # when contains 'LEAD:' it may be repeating the first paragraph
                if s.find('LEAD:') == -1:
                    full_text += '\n' + s

    return docid, category, full_text


def readXMLs(path = '../data_slice', dest = '../data_pickle'):
    """

    :param path: xml dir
    :param dest: save path
    :return: # of parsed docs

    I save each parsed doc in a separate pickle file in 'dest' for later use
    """
    if not os.path.exists(dest):
        os.mkdir(dest)
    relative = []
    for root, dirs, files in os.walk(path):
        relative = [x for x in files if 'xml' in x]

    cnt = 0
    for f in relative:
        newf = str(f).replace('xml', 'p')
        docid, category, full_text = readSingleXML(os.path.join(path, f))
        # filter out no tag or no text news
        if len(category) == 0 or len(full_text) == 0:
            continue
        cnt = cnt + 1
        if cnt % 1000 == 0:
            print cnt
        pickle.dump((docid, category, full_text), open(os.path.join(dest, newf), 'wb'))

    return cnt


def printXMLs(path = '../data_pickle', num = 100):
    """

    :param path: pickle doc dir path
    :param num: max # to be print
    :return: None

    Print the doc from parsed pickle file
    """
    relative=[]
    for root, dirs, files in os.walk(path):
        relative = [x for x in files if 'p' in x]

    random.shuffle(relative)

    for f in relative[:num]:
        d = pickle.load(open(os.path.join(path, f)))
        print d


def extractCategories(path = '../data_pickle', dest = '../data_category', freqThreshold = 20):
    """

    :param path: path of pickle doc
    :param dest: save path for category extraction answer
    :param freqThreshold: frequency threshold for selecting categories
    :return: selected - a list of selected categoires, dump to 'category.p'; allMap - a category-frequency map, dump to 'all.p'
    """
    f = []
    for root, dirs, files in os.walk(path):
        f = [x for x in files if 'p' in x]
    allMap = {}
    cnt = 0
    for nowf in f:
        d = pickle.load(open(os.path.join(path, nowf)))
        for c in d[1]:
            if c in allMap:
                allMap[c] = allMap[c] + 1
            else:
                allMap[c] = 1
        cnt = cnt + 1
        if cnt % 1000 == 0:
            print cnt
    selected = [x for x in allMap.keys() if allMap[x] >= freqThreshold]
    if not os.path.exists(dest):
        os.mkdir(dest)
    pickle.dump(selected, open(os.path.join(dest, 'category.p'), 'wb'))
    pickle.dump(allMap, open(os.path.join(dest, 'all.p'), 'wb'))
    return selected, allMap


def extractWordBagVector(path = '../data_pickle', dest = '../data_bagvec', num = 100000):
    """

    :param path: path of pickle doc
    :param dest: save path for doc words bag representation
    :param num: the max # of processing
    :return: # processed
    """
    if not os.path.exists(dest):
        os.mkdir(dest)
    f = []
    for root, dirs, files in os.walk(path):
        f = [x for x in files if 'p' in x]
    cnt = 0
    for nowf in f[:num]:
        d = pickle.load(open(os.path.join(path, nowf)))
        text = d[2]
        # break into sentences
        oldSentences = nltk.sent_tokenize(text)
        identity = string.maketrans('-', ' ')
        sentences = []
        for s in oldSentences:
            s = string.translate(s, identity)
            sentences.append(string.translate(s, identity, string.punctuation + string.digits))
        # break into words
        words = []
        for s in sentences:
            words.extend(nltk.tokenize.word_tokenize(s))
        # spelling check
        dict = enchant.Dict("en_US")
        words1 = [w for w in words if dict.check(w)]
        # delete short words and stop words
        words2 = [w.lower() for w in words1 if w.lower() not in nltk.corpus.stopwords.words('english') and len(w) >= 3]
        # word stem
        porter = nltk.stem.SnowballStemmer('english')
        words3 = [porter.stem(w) for w in words2]
        # delete short words and stop words again
        words4 = [w for w in words3 if w not in nltk.corpus.stopwords.words('english') and len(w) >= 3]
        # trans to dict
        dic = {}
        for w in words4:
            if w in dic:
                dic[w] = dic[w]+1
            else:
                dic[w] = 1
        # final list
        ans = [d[0], d[1], dic]
        pickle.dump(ans, open(os.path.join(dest, nowf), 'wb'))

        cnt = cnt + 1
        if cnt % 10 == 0:
            print cnt
    return cnt


def printWordBags(path = '../data_bagvec', num = 10):
    """

    :param path: path of document vectors
    :param num: max # for print
    :return: None
    """
    f = []
    for root, dirs, files in os.walk(path):
        f = [x for x in files if 'p' in x]

    random.shuffle(f)

    for nowf in f[:num]:
        d = pickle.load(open(os.path.join(path, nowf)))
        print d


def getWordDict(path = '../data_bagvec', dest = '../data_dict', threshold = 1):
    """

    :param path: path of document vectors
    :param dest: path to save the word dictionary
    :param threshold: only words repeating more or equal then threshold are added to word list
    :return: word list: [[word, # of appeared docs], ...]
    """
    if not os.path.exists(dest):
        os.mkdir(dest)
    wordList = []
    wordDict = {}
    wordRepeatCnt = {}
    f = []
    for root, dirs, files in os.walk(path):
        f = [x for x in files if 'p' in x]

    cnt = 0
    for nowf in f:
        d = pickle.load(open(os.path.join(path, nowf)))
        for w in d[2]:
            if w in wordDict:
                wordList[wordDict[w]][1] = wordList[wordDict[w]][1] + 1
                wordRepeatCnt[w] = wordRepeatCnt[w] + d[2][w]
            else:
                wordList.append([w, 1])
                wordDict[w] = len(wordList) - 1
                wordRepeatCnt[w] = d[2][w]

        cnt = cnt + 1
        if cnt % 100 == 0:
            print cnt
    newWordList = [x for x in wordList if wordRepeatCnt[x[0]] >= threshold]
    pickle.dump(newWordList, open(os.path.join(dest, 'wordList.p'), 'wb'))
    return newWordList

def genTfIdfVecs(bagPath = '../data_bagvec', dictPath = '../data_dict/wordList.p', savePath = '../data_tfidf', fold = 10):
    """

    :param bagPath: document word bag representation folder path
    :param dictPath: word dictionary file path
    :param savePath: place to save the Tf-Idf representation
    :param fold: Evenly divide the document set to # folds for cross-validation
    :return: None
    savePath + 'label#.p' - [[no, [categories]], [no, [categories]], ...]
    savePath + 'vec#.npy' - data:[tf-idf0, tf-idf1, ...; tf-idf0, tf-idf1, ...;] (a numpy matrix, each line is a document) no:[doc-id0, doc-id1, ...]
    """

    if not os.path.exists(savePath):
        os.mkdir(savePath)

    # build index for word dictionary
    wordList = pickle.load(open(dictPath))
    wordIndex = {}
    for i in range(0, len(wordList)):
        wordIndex[wordList[i][0]] = i

    # get file list & shuffle it
    allf = []
    for root, dirs, files in os.walk(bagPath):
        allf = [x for x in files if 'p' in x]
    random.shuffle(allf)

    # calculate # of each fold
    foldN = int(math.ceil(float(len(allf)) / float(fold)))

    for i in range(0, fold):
        # get slice for current fold
        cnt = 0
        print "Fold %d" % (i)
        stNo = i * foldN
        nowf = allf[stNo:stNo+foldN]
        categoryList = []
        mat = np.zeros((len(nowf),len(wordList)), float)
        noMat = np.zeros((len(nowf)), int)
        for f in nowf:
            no, category, vec = genTfIdfVec(len(allf), wordList, wordIndex, pickle.load(open(os.path.join(bagPath, f))))
            categoryList.append([no, category])
            mat[cnt, :] = vec
            noMat[cnt] = no
            cnt = cnt + 1
            if (cnt % 10 == 0):
                print cnt
        categoryPath = "label%d.p" % (i)
        vecPath = "vec%d.npz" % (i)
        pickle.dump(categoryList, open(os.path.join(savePath, categoryPath), 'wb'))
        np.savez(open(os.path.join(savePath, vecPath), 'wb'), data = mat, no = noMat)


def genTfIdfVec(totDoc, wordList, wordIndex, wordBag):
    """
    Function called by genTfIdfVecs() to generate tf-idf vector for a single doc
    :param totDoc: # of docs
    :param wordList: the word list read from data_dict folder
    :param wordIndex: the map used for spotting word in wordList
    :param wordBag: wordbag of the single doc, directly obtained form data_bagvec
    :return: no - docid, category - a list of category tags, arr - a numpy vector, the first element is doc ID, proceeding by tf-idf vector
    """
    no = wordBag[0]
    category = wordBag[1]
    doc = wordBag[2]
    totWord = sum([doc[x] for x in doc.keys()])
    arr = np.zeros((len(wordList)), float)
    for w in doc:
        if w in wordIndex:
            t = doc[w]
            ind = wordIndex[w]
            tf = float(t) / float(totWord)
            idf = math.log(float(totDoc) / float(wordList[ind][1]))
            arr[ind] = tf * idf
    return no, category, arr

def preprocess():
    """
        Preprocess summary func
    :return: None
    """
    # parse XML -> pickle

    t0 = time.clock()
    print "# of parsed docs: %d" % (readXMLs('../samples_50000'))
    t1 = time.clock()
    print "Parsing seconds: %f" % (t1 - t0)

    # random print some pickles

    printXMLs()

    # extract categories

    t2 = time.clock()
    selected, allMap = extractCategories(freqThreshold=500)
    print "Tot category: %d" % (len(selected))
    t3 = time.clock()
    print "Extract categories seconds: %f" % (t3 - t2)


    # transform pickles to wordbag maps
    t4 = time.clock()
    extractWordBagVector()
    t5 = time.clock()
    print 'Extract wordBag vector seconds: %f' % (t5 - t4)


    # print wordbags
    printWordBags(num = 100)


    # retrieve dictionary
    t6 = time.clock()
    wordList = getWordDict(threshold = 10)
    t7 = time.clock()
    print 'Extract word dictionary seconds: %f' % (t7 - t6)


    # generate tf-idf vector
    t8 = time.clock()
    genTfIdfVecs()
    t9 = time.clock()
    print 'Generate Tf-Idf vector seconds: %f' % (t9 - t8)


