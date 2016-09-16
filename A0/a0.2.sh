#!/bin/bash 
cut -f 3 cmudict.tsv | sort | uniq -c | egrep -v '\s1'

