# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ENLP A1 Part I: Naive Bayes

Usage: python nbmodel.py alpha | python nbmodel.py -l alpha

(Adapted from Alan Ritter)
"""
import sys, os, glob

import numpy as np
from collections import Counter
from math import log
from math import exp

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize  #word_tokenize(line)

from evaluation import Eval

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""

    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    # open the file at file_path, construct a list of its word tokens,
    # and append that list to 'docs'.
    # look up the document's label and append it to 'labels'.
    docs, labels = [], []
    wnl = WordNetLemmatizer()
    if lemmatize == False:
        for file_path in glob.glob(os.path.join(direc, '*.txt')):
            filename = os.path.basename(file_path)
            doc = []
            with open(file_path) as f:
                for ln in f.readlines():
                    line_token = ln.strip().split(' ')
                    for word in line_token:
                        if word.isalnum():
                            doc.append(word)
                docs.append(doc)
            labels.append(labelMap[filename])
    else:
        for file_path in glob.glob(os.path.join(direc, '*.txt')):
            filename = os.path.basename(file_path)
            doc = []
            with open(file_path) as f:
                for ln in f.readlines():
                    line_token = ln.strip().split(' ')
                    for word in line_token:
                        if word.isalnum():
                            doc.append(wnl.lemmatize(word))
                docs.append(doc)
            labels.append(labelMap[filename])

    return docs, labels

class NaiveBayes:
    def __init__(self, train_docs, train_labels, ALPHA):
        # list of native language codes in the corpus
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']

        self.ALPHA=ALPHA
        self.priorProbs = {l: 0 for l in self.CLASSES}
        self.likelihoodProbs = {l: Counter() for l in self.CLASSES}
        self.trainVocab = set()
        self.learn(train_docs, train_labels, alpha=self.ALPHA)

    def learn(self, docs, labels, alpha):
        """Estimate parameters for a naive Bayes bag-of-words model with the
        given training data and amount of add-alpha smoothing."""

        assert len(docs)==len(labels)
        labelCounts = {l: 0 for l in self.CLASSES}
        wordCounts = {l: Counter() for l in self.CLASSES}
        totalWordCounts = {l: 0 for l in self.CLASSES}

        # iterate over documents in order to record, count(y) in labelCounts,
        for label in labels:
            labelCounts[label] += 1
        
        # count(y,word) in wordCounts,
        for i, doc in enumerate(docs):
            for j in doc:
                wordCounts[labels[i]][j] += 1

        # count(y,w) for all words in totalWordCounts,
        for y in wordCounts:
            for word in wordCounts[y]:
                totalWordCounts[y] += wordCounts[y][word]

        # and to store the training vocabulary in self.trainVocab
        for y in wordCounts:
            self.trainVocab.update(list(wordCounts[y].keys()))

        # compute and store prior distribution over classes
        # (unsmoothed) in self.priorProbs
        docsLength = len(docs)
        for y in labelCounts:
            self.priorProbs[y] = (1.0*labelCounts[y] / docsLength)

        # compute and store p(w|y), with add-alpha smoothing,
        V = len(self.trainVocab)
        for y in wordCounts:
            for word in self.trainVocab: #for size of V
                if word in wordCounts[y]:
                    self.likelihoodProbs[y][word] = (wordCounts[y][word] + alpha) / (totalWordCounts[y] + alpha * (V + 1))
                else:
                    self.likelihoodProbs[y][word] = alpha / (totalWordCounts[y] + alpha * (V + 1))


        # in self.likelihoodProbs. Add '**OOV**' as a pseudo-word
        # for out-of-vocabulary items (but do not include it in self.trainVocab).
        for y in self.priorProbs:
            self.likelihoodProbs[y]['**OOV**'] = alpha / (totalWordCounts[y] + alpha * (V + 1))

        # Sanity checks--do not modify
        assert len(self.priorProbs)==len(self.likelihoodProbs)==len(self.CLASSES)>2
        assert .999 < sum(self.priorProbs.values()) < 1.001
        for y in self.CLASSES:
            assert .999 < sum(self.likelihoodProbs[y].values()) < 1.001,sum(self.likelihoodProbs[y].values())
            assert 0 <= self.likelihoodProbs[y]['**OOV**'] < 1.0,self.likelihoodProbs[y]['**OOV**']

    def joint_prob(self, doc, y):
        # compute the log of the joint probability of the document and the class,
        # i.e., return p(y)*p(w1|y)*p(w2|y)*... (but in log domain)
        # should not make any changes to the model parameters
        prior = self.priorProbs[y]
        joint = 1.0 * log(prior)
        oov_lli = log(self.likelihoodProbs[y]['**OOV**'])
        for word in doc:
            if word in self.trainVocab:
                lli = log(self.likelihoodProbs[y][word])
                joint += lli
            else:
                joint += oov_lli

        return joint

    def predict(self, doc):
        # apply Bayes' rule: return the class that maximizes the
        # prior * likelihood probability of the test document
        # should not make any changes to the model parameters
        res = [self.joint_prob(doc,y) for y in self.CLASSES]
        pred = max(res)
        for i, num in enumerate(res):
            if num == pred:
                return self.CLASSES[i]

    def eval(self, test_docs, test_labels):
        """Evaluates performance on the given evaluation data."""
        assert len(test_docs)==len(test_labels)
        preds = []  # predicted labels
        for doc,y_gold in zip(test_docs,test_labels):
            y_pred = self.predict(doc)
            preds.append(y_pred)

        ev = Eval(test_labels, preds)
        for y in self.priorProbs:
            print ((y, self.priorProbs[y]))
        return ev.accuracy()

if __name__ == "__main__":
    lemmatize = False
    args = sys.argv[1:]
    if args[0] == '-l':
        lemmatize = True
        args = args[1:]
    alpha = float(args[0])

    #alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    #accs = []

    train_docs, train_labels = load_docs('train', lemmatize)
    print(len(train_docs), 'training docs with', sum(len(d) for d in train_docs), 'tokens', file=sys.stderr)
    test_docs,  test_labels  = load_docs('test', lemmatize)
    print(len(test_docs), 'eval docs with', sum(len(d) for d in test_docs), 'tokens', file=sys.stderr)


    nb = NaiveBayes(train_docs, train_labels, alpha)
    acc = nb.eval(test_docs, test_labels)

    print (acc)
