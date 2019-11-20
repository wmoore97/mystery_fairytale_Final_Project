#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:37:55 2019

@author: chengchen
"""
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# The process of extracting out the root word is called stemming.

import string
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt


def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    stop_words.add('havent')
    stop_words.add('could')
    
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    # Perform stemming on the tokenized words 
    tokens = [stemmer.stem(x) for x in tokens]
    
    return tokens

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def getUniqueWords(line2) :
    uniqueWords = [] 
    for i in line2:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords

# Gather all of the domuments in to result.txt
import glob
path = "."
filename = 'result.txt'
stemmer = SnowballStemmer("english")

from operator import itemgetter, attrgetter
import operator

with open("result.txt", "wb") as outfile:
    for filename in glob.iglob(path+"/data/fairytale/*.txt"):
         with open(filename, "rb") as f:
             outfile.write(f.read())
             text = load_doc(filename)
             text = text.lower() # Convert all to lower case
             line2 = clean_doc(text) # Combine the words by the roots
             # Count the frequency of each word
             wubFD = FreqDist(word for word in line2)

            # Sort the filtered words
#             sorted(wubFD.items(), key=lambda kv: kv[0,1], reverse=True)
             XY = wubFD.items()
             wubFD = sorted(wubFD.items(), key=operator.itemgetter(1), reverse=True)

             # Plot the each word with corresponding to its frequency             
#             limit = len(wubFD)
#             X = [x for (x,y) in wubFD[:limit]]
#             Y = [y for (x,y) in wubFD[:limit]]
#             nX = range(len(X))
#             offset = (plt.gca().get_xticks()[1] - plt.gca().get_xticks()[0]) / 10
#             plt.scatter(nX, Y)
##             plt.xticks(nX, X, rotation='vertical', size=10)    
#             for (pt, label) in enumerate(X):
#                 plt.annotate(label,
#                 xy=(nX[pt], Y[pt]),
#                 xytext=(nX[pt]+offset, Y[pt]),size=5)                      
#             plt.xlabel('Tokens')
#             plt.ylabel('Counts')
#             plt.title('Counts by tokens')
#             plt.show()

# Filtered words
#print(line2)

# The number of unique words
#print(len(getUniqueWords(line2)))

# Create bag-of-word matrix
            
