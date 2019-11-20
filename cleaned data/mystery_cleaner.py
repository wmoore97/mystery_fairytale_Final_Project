# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:13:16 2019

e@author: Moore

This script was designed to trim the header and footer info out of each ebook in teh data set

"""

import glob
import nltk as nl
import string
from nltk.stem.snowball import SnowballStemmer

#nl.download('stopwords')

path = "."



# Replace "mystery" for whichever folder of txt files you want to proccess, and make sure trimmer.py is in the paretn directory
for filename in glob.iglob(path+"/mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        table = str.maketrans('', '', string.punctuation)
        for punc in lines:
            punc.translate(table)
        token_text = nl.tokenize.word_tokenize(lines)
        stop_words = set(nl.corpus.stopwords.words("english"))
        stop_words.add('havent')
        stop_words.add('could')
        filtered_data=[]
        for w in token_text:
            if w not in stop_words:
                filtered_data.append(w)
        filtered_data2 =[]
        for word in filtered_data:
            if word.isalpha():
                filtered_data2.append(word)
        filtered_data3=[]
        for words in filtered_data2:
            if len(words) > 1:
                filtered_data3.append(words)
        filtered_data4=[]
        stemmer = SnowballStemmer("english")
        for x in filtered_data3:
            filtered_data4.append(stemmer.stem(x))
        f.close()
        o = open(filename, "w", encoding="utf8")
        o.write(str(filtered_data4))
        o.close()