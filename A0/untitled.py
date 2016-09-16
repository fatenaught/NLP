#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import json, re, sys, doctest, collections

"""
ENLP A0.5-6: Given an English word, list its rhymes
based on the CMU Pronouncing Dictionary.
"""
'''
def load_dict():
    """Load cmudict.json into the CMUDICT dict."""
    data = []
    global CMUDICT 
    CMUDICT= collections.OrderedDict()  # word -> pronunciations
    INPUT_PATH = 'cmudict.json'
    with open(INPUT_PATH) as datafile:    
        for row in datafile:
            row = row.replace('\n','')
            data.append(json.loads(row))
    
    #get new dictionary 
    for i in data:
        CMUDICT[i['word']] = i['pronunciations']

def pronunciations(word):
    """Get the list of possible pronunciations for the given word
    (as a list of lists) by looking the word up in CMUDICT.
    If the word is not in the dictionary, return None.

    TODO: write a few doctests below.

    >>> pronunciations('A')
    [['AH0'], ['EY1']]

    >>> pronunciations('ZYWICKI')
    [['Z', 'IH0', 'W', 'IH1', 'K', 'IY0']]
    """
    word = word.upper()
    if word not in list(CMUDICT.keys()):
        return 'Query word not found'
    else:
        return CMUDICT[word]
     

def rhyming_words(word):
    """Get the list of words that have a pronunciation
    that rhymes with some pronunciation of the given word.

    >>> 'STEW' in rhyming_words('GREW')
    True

    >>> 'GROW' in rhyming_words('GREW')
    False

    >>> 'GREW' in rhyming_words('GREW')
    True

    TODO: write more doctests
    >>> 'ZYLKA' in rhyming_words('A')
    True
    """

    #initilization
    word = word.upper()
    rhymes = collections.OrderedDict()
    res = []
    last = []

    #build a rhyme dictionary
    for i in list(CMUDICT.keys()):
        rhymes[i] = []
        for j in CMUDICT[i]:
            rhymes[i].append(j[-1])

    #find rhymes for the word
    for j in rhymes[word]:
        last.append(j)

    #compare the input_word to with the rhyme dictionary
    for i in last:
        for j in rhymes:
            if i in rhymes[j]:
                res.append(j)
    return res

if __name__=='__main__':
    #if execute like: python rhymes.py 'A'
    if len(sys.argv) == 2:
        input_word = sys.argv[1]
        load_dict()
        doctest.testmod()
        if pronunciations(input_word) != 'Query word not found':
            print(rhyming_words(input_word))
        else:
            print('Query word not found')

    #if execute like: python rhymes.py -p 'A'
    if len(sys.argv) == 3:
        input_word_2 = sys.argv[2]
        load_dict()
        new_dict = {}
        arpabet = ['AO', 'AA', 'IY', 'UW','EH', 'IH', 'UH','AH','AX', 'AE', 'EY', 'AY','OW','AW','OY', 'ER', 'AXR','EH R', 'UH R', 'AO R', 'AA R', 'IH R', 'IY R', 'AW R']
        symbol = ['ɔ', 'ɑ', 'i', 'u', 'ɛ', 'ɪ', 'ʊ', 'ʌ', 'ə', 'æ', 'eɪ', 'aɪ', 'oʊ','aʊ', 'ɔɪ', 'ɝ', 'ɚ', 'ɛr', 'ʊr', 'ɔr', 'ɑr', 'ɪr', 'ɪr', 'aʊr']
        for i in range(len(arpabet)):
            new_dict[arpabet[i]] = symbol[i]
        p = pronunciations(input_word_2)

        new_pronun = []
        for i in p: #[u'Z', u'IH0', u'W', u'IH1', u'K', u'IY0']
            new_pronun_ele = []
            for j in i:
                if j not in arpabet:
                    new_pronun_ele.append(j)
                else:
                    new_pronun_ele.append(new_dict[j])
            new_pronun.append(' '.join(new_pronun_ele))
        print (new_pronun)


if p != 'Query word not found':
    words = ['A','APPLE']
    
    pronuns = []
    for i in words:
        pronuns.append(pronunciations(i)[0])

    new_pronun = []
    for i in pronuns: #[u'Z', u'IH0', u'W', u'IH1', u'K', u'IY0']
        new_pronun_ele = []
        for j in i:
            if j.isalpha():
                new_pronun_ele.append(j)
            else:
                if j[:-1] in list(new_dict.keys()):
                    new_pronun_ele.append(new_dict[j][:-1]+j[-1])
                else:
                    new_pronun_ele.append(j)
        new_pronun.append(' '.join(new_pronun_ele))
    print (new_pronun)

else:
    print ('Query word not found')
'''

data = []
CMUDICT= collections.OrderedDict()  # word -> pronunciations
INPUT_PATH = 'cmudict.json'
with open(INPUT_PATH) as datafile:    
    for row in datafile:
        row = row.replace('\n','')
        data.append(json.loads(row))

#get new dictionary 
for i in data:
    CMUDICT[i['word']] = i['pronunciations']

rhymes = collections.OrderedDict()
res = []
last = []

#build a rhyme dictionary
for i in list(CMUDICT.keys()):
    rhymes[i] = []
    for j in CMUDICT[i]:
        for z in j[::-1]:
            if not z.isalpha():
                rhymes[i].append(z[:-1])
                break
print rhymes










