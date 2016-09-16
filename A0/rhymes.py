#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
ENLP A0.5-6: Given an English word, list its rhymes
based on the CMU Pronouncing Dictionary.

what counts as a rhyme: the last vowel (syllable with stress level) matches.
"""

import json, re, sys, doctest, collections

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
            for z in j[::-1]:
                if not z.isalpha():
                    rhymes[i].append(z[:-1])
                    break

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
            print((rhyming_words(input_word)))
        else:
            print('Query word not found')

    #if execute like: python rhymes.py -p 'A'
    if len(sys.argv) == 3:
        input_word_2 = sys.argv[2]
        load_dict()
        new_dict = {}
        arpabet = ['AO', 'AA', 'IY', 'UW','EH', 'IH', 'UH','AH','AX', 'AE', 'EY', 'AY','OW','AW','OY']
        symbol = ['ɔ', 'ɑ', 'i', 'u', 'ɛ', 'ɪ', 'ʊ', 'ʌ', 'ə', 'æ', 'eɪ', 'aɪ', 'oʊ','aʊ', 'ɔɪ']
        for i in range(len(arpabet)):
            new_dict[arpabet[i]] = symbol[i]

        p = pronunciations(input_word_2)
        #[['B', 'AH0', 'CH', 'IH1', 'N', 'S', 'K', 'IY0']]

        if p != 'Query word not found':
            words = rhyming_words(input_word_2)

            pronuns = []
            for i in words:
                pronuns.append(pronunciations(i))
            #pronuns:[[u'Z', u'IH0', u'W', u'IH1', u'K', u'IY0'],[[u'A],[u'B]]]
            #new_pronun:[[u'Z', u'ɪ0', u'W', u'ɪ1', u'K', u'u0'],[[u'A],[u'B]]]            

            large_pronun = []
            for i in pronuns: #[[u'A],[u'B]]
                new_pronun = []
                for j in i: #[u'A]
                    new_pronun_ele = []
                    for z in j:
                        if z.isalpha():
                            new_pronun_ele.append(z)
                        else:
                            vow = z[:-1]
                            if vow in list(new_dict.keys()):
                                new_pronun_ele.append(new_dict[vow]+z[-1])
                            else:
                                new_pronun_ele.append(z)
                    new_pronun.append(' '.join(new_pronun_ele))
                large_pronun.append(new_pronun)
            print (large_pronun)

        else:
            print ('Query word not found')

        







