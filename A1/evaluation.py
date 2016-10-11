import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

class Eval:
    def __init__(self, gold, pred):
        assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred
        self.labels = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']

    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred,self.gold) if p==g)
        return float(numer) / len(self.gold)

    def c_matrix(self):
        print('confusion matrix')
        y_actu = pd.Series(self.gold, name='Actual')
        y_pred = pd.Series(self.pred, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        return df_confusion

    def precision(self):
        print('precision')
        Precision = (precision_score(self.gold, self.pred, average=None, labels = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']))
        for i in range(0, len(self.labels)):
            print(self.labels[i],Precision[i])

    def recall(self):
        print('recall')
        Recall = (recall_score(self.gold, self.pred, average=None, labels = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']))
        for i in range(0, len(self.labels)):
            print(self.labels[i],Recall[i])

    def f1(self):
        print('F1')
        F1 = (f1_score(self.gold, self.pred, average=None, labels = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']))
        for i in range(0, len(self.labels)):
            print(self.labels[i],F1[i])



