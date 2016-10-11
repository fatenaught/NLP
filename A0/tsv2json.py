#!/usr/bin/env python
# -*- coding: utf-8 -*- 

'''
ENLP A0.4
converts cmudict.tsv to cmudict.json
'''

import csv
import json
import collections

#initialization
dictraw = []
cmudict = collections.OrderedDict()

#open file and read in 
with open('cmudict.tsv') as tsvfile: #path
#with open('cmudict.tsv', encoding='utf-8', errors='ignore') as tsvfile: #path
    csvData = csv.reader(tsvfile, delimiter = '\t') #readin
    for row in csvData:
        dictraw.append(row)
tsvfile.close

#append to a dictionary
for i in range(len(dictraw)):
    #if wordlist[i] not in dictfile.keys():
    if dictraw[i][1] == '':
        pronun = []
        pronun.append(dictraw[i][2].split())
        cmudict[dictraw[i][0]] = pronun
    else:
        cmudict[dictraw[i][0]].append(dictraw[i][2].split())

#dump to json file
with open('cmudict.json', 'w') as outfile:
    for i in cmudict:
        resline = {}
        resline['word'] = i
        resline['pronunciations'] = cmudict[i]
        json.dump(resline, outfile)
        outfile.write('\n')
outfile.close()


