#!/bin/bash
egrep -i '^[A-Z]+' cmudict-0.7b | sed -r 's/\s\s/\t\t/g'| sed -r 's/\(/\t\(/g' | sed -r 's/\)\s\s/\)\t/g' > cmudict.tsv