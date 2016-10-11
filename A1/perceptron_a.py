#!/usr/bin/env python3
"""
ENLP A1 Part II: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob
import operator

from collections import Counter
import heapq
from math import log
from numpy import mean

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from evaluation import Eval

from nbmodel import load_docs, NaiveBayes

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    #Add bias
    ff['ajdif7af'] = 1

    wnl = WordNetLemmatizer()

    
    #Unigram
    for word in set(doc):
        ff[wnl.lemmatize(word)] = 1
    '''
    #2-gram
    for i in range(len(doc)-1):
        ff[doc[i] + ' ' + doc[i+1]] = 1
    

    #Lemma
    for word in set(doc):
        ff[wnl.lemmatize(word)] = 1
    '''
    
    #Lemma 2-gram
    for i in range(len(doc)-1):
        ff[wnl.lemmatize(doc[i]) + ' ' + wnl.lemmatize(doc[i+1])] = 1

    
    #upper-normalization
    for word in set(doc):
        ff[word.lower()] = 1

    return ff

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels


class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS, dev_docs, dev_labels, flag):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        #self.weights = {l: {word:1 for word in trainVocab} for l in self.CLASSES}
        self.weights = {l: Counter() for l in self.CLASSES}
        self.a_weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels, flag)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels, flag):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        if flag == False:
            best_weights = {}
            best_acc = 0
            
            for i in range(self.MAX_ITERATIONS):
                flag = [False] * len(train_docs)
                count = 0
                for j, doc in enumerate(train_docs):
                    pred = self.predict(doc)
                    gold = train_labels[j]
                    if pred != gold:
                        count += 1
                        for word in list(doc.keys()):
                            self.weights[gold][word] += 0.1 * doc[word]
                            self.weights[pred][word] -= 0.1 * doc[word]
                    else:
                        flag[j] = True

                train_acc = self.test_eval(train_docs, train_labels)
                test_acc = self.test_eval(self.dev_docs, self.dev_labels)
                print ('iteration: ' + str(i) + ' updates=' + str(count) + ', trainAcc=' + str(train_acc) +', devAcc=' + str(test_acc) + ', params=' + str(sum(len(v) for v in self.weights.values())))

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_weights = self.copy_weights()

                if count == 0:
                    break
                else:
                    if False in flag:
                        continue
                    else:
                        break
            self.weights = best_weights

        else:
            {'HIN':{'a':2, 'b':3}, 'JPN':{'a':3, 'b':4}}
            a_weights = {l: Counter() for l in self.CLASSES}
            
            for i in range(self.MAX_ITERATIONS):
                flag = [False] * len(train_docs)
                count = 0
                for j, doc in enumerate(train_docs):
                    pred = self.predict(doc)
                    gold = train_labels[j]
                    if pred != gold:
                        count += 1
                        for word in list(doc.keys()):
                            self.weights[gold][word] += 0.1 * doc[word]
                            self.weights[pred][word] -= 0.1 * doc[word]

                            a.weights[gold][word] += 0.1 * (i+1) * doc[word]
                            a.weights[pred][word] -= 0.1 * (i+1) * doc[word]

                        self.weights[pred]['ajdif7af'] += 0.1 * self.weights[pred]['ajdif7af']
                        a.weights[pred]['ajdif7af'] += 0.1 * (i+1) * a.weights[pred]['ajdif7af']

                    else:
                        flag[j] = True

                train_acc = self.test_eval(train_docs, train_labels)
                test_acc = self.test_eval(self.dev_docs, self.dev_labels)
                print ('iteration: ' + str(i) + ' updates=' + str(count) + ', trainAcc=' + str(train_acc) +', devAcc=' + str(test_acc) + ', params=' + str(sum(len(v) for v in self.weights.values())))

                if count == 0:
                    break
                else:
                    if False in flag:
                        continue
                    else:
                        break

            for y in self.weights:
                for word in y:
                    self.weights[y][word] = self.weights[y][word] - (1/i)*a.weights[y][word]

        #error analysis
        for y in self.weights.keys():
            print (y)
            top10 = heapq.nlargest(10, self.weights[y], key=self.weights[y].get)
            r_top10 = heapq.nsmallest(10, self.weights[y], key=self.weights[y].get)

            print('Highest')
            for word in top10:
                print(word, self.weights[y][word])

            print('Lowest')
            for word in r_top10:
                print(word, self.weights[y][word])
            
            print('Biased', self.weights[y]['ajdif7af'])


    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        score = 0
        for word in list(doc.keys()):
            score += doc[word] * (self.weights[label][word])
        return score

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        scores = []
        for y in self.CLASSES:
            scores.append((self.score(doc, y), y))
        pred = sorted(scores)
        return pred[-1][1]

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        #print (ev.c_matrix())
        ev.precision()
        ev.recall()
        ev.f1()
        return ev.accuracy()


if __name__ == "__main__":
    flag = False
    args = sys.argv[1:]
    if args[0] == '-a':
        flag = True
        args = args[1:]
    niters = args[0]

    train_docs, train_labels = load_featurized_docs('train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)

    test_docs,  test_labels  = load_featurized_docs('test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels, flag)
    acc = ptron.test_eval(test_docs, test_labels)
    print(acc, file=sys.stderr)

    '''
    top_10 = []
    r_top_10 = []
    bias = []
    for y in ptron.weights:
        t10 = []
        r_t10 = []
        sorted_x = sorted(ptron.weights[y].items(), key=operator.itemgetter(1))
        t10.append(sorted_x[-1:-11:-1])
        r_t10.append(sorted_x[:10])
        top_10.append((t10,y))
        r_top_10.append((r_t10,y))
    print (top_10, r_top_10, bias)
    '''
    

