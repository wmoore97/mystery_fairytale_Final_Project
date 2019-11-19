# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:13:16 2019

e@author: Moore

This script was designed to trim the header and footer info out of each ebook in teh data set

"""

import glob
import nltk as nl

#nl.download('stopwords')

path = "."



# Replace "mystery" for whichever folder of txt files you want to proccess, and make sure trimmer.py is in the paretn directory
for filename in glob.iglob(path+"/mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        token_text = nl.tokenize.word_tokenize(lines)
        stop_words = set(nl.corpus.stopwords.words("english"))
        filtered_data=[]
        for w in token_text:
            if w not in stop_words:
                filtered_data.append(w)
        f.close()
        o = open(filename, "w", encoding="utf8")
        o.write(str(filtered_data))
        o.close()