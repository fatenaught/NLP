import sys, os, glob
from collections import Counter
from math import log
from nltk.stem.wordnet import WordNetLemmatizer
import pandas

language = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
count = ['51', '34', '53', '47', '53', '60', '60', '52', '62', '57', '69']

alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
acc = [0.705685618729097, 0.7190635451505016, 0.7207357859531772, 0.7190635451505016, 0.6956521739130435, 0.6755852842809364, 0.5852842809364549, 0.3979933110367893]

d1=pandas.DataFrame(alphas)
d1.columns = ['alpha']
d2=pandas.DataFrame(acc)
d2.columns = ['accuracy']

merg1=d1.join(d2)
merg1.to_csv('acc_count.csv',sep = ',', encoding = 'utf-8')
