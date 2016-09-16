#!/bin/bash
cut -f 1,3 cmudict.tsv | grep -E '[0-9].+[0-9]' | cut -f 1
#syllable: vowels = pronunciation with stress value
#Polysyllabic: multiple (>= 2) syllabes in the pronunciation

#LIEUTENANT: ["L", "UW0", "T", "EH1", "N", "AH0", "N", "T"]; 2 syllables
#TUITION: ["T", "Y", "UW0", "IH1", "SH", "AH0", "N"]; 3 syllables
#CHOIRS: ["K", "W", "AY1", "R", "Z"]; 1 syllable
#FAMILY: ["F", "AE1", "M", "AH0", "L", "IY0"], ["F", "AE1", "M", "L", "IY0"]; 3/2 syllables
#INTERESTING: ["IH1", "N", "T", "AH0", "R", "EH2", "S", "T", "IH0", "NG"]; 4 syllables
#NORMALLY: ["N", "AO1", "R", "M", "AH0", "L", "IY0"], ["N", "AO1", "R", "M", "L", "IY0"]: 3/2 syllables





#'[0-9]\{2,\}'
