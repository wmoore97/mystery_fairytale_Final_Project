#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:01:58 2019

@author: chengchen
"""
from __future__ import print_function
import sklearn
import numpy as np
# Import all of the scikit learn stuff
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import seaborn as sns

path = "."
documents = []

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
    stop_words.add('öfver')
    stop_words.add('êtes')
    
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    # Perform stemming on the tokenized words 
    #    tokens = [stemmer.stem(x) for x in tokens]
    return tokens

def convertTuple(tup): 
    str =  ''.join(tup) 
    return str

#for filename in glob.iglob(path+"/fairytale/*.txt"):
#    #iterate through each file in the folder, opening it
#    with open(filename, encoding="utf8") as f:
#        lines = f.read()
#        lines = lines.lower() # Convert all to lower case
##        lines = clean_doc(lines) # Combine the words by the roots
#        documents.append(lines)
#        f.close()
        
count = 0
for filename in glob.iglob(path+"/mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        lines = lines.lower() # Convert all to lower case
#        lines = clean_doc(lines) # Combine the words by the roots
        f.close()
        count+=1
        documents.append(lines)
#        print(documents)
#        if count ==15:
#            break

# Restore all document names
all_files = os.listdir("./mystery")

documents = np.array(documents)
#documents = convertTuple(documents)
print(documents)
#exit()

#for i in range documents.shape:
example = documents

# TF-IDF as input
vectorizer = TfidfVectorizer()
#vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
dtm = vectorizer.fit_transform(example)

pd.DataFrame(dtm.toarray(),index=all_files,columns=vectorizer.get_feature_names()).head(10)
#print(pd.DataFrame(dtm.toarray(),index=all_files,columns=vectorizer.get_feature_names()).head(10))

## Fit LSA. Use algorithm = “randomized” for large datasets
lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm = dtm.asfptype()
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names())
#print(pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names()))
#exit()


xs = [w[0] for w in dtm_lsa]
ys = [w[1] for w in dtm_lsa]
xs, ys
plt.scatter(xs,ys)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')
plt.show()


#import matplotlib.pyplot as plt
plt.figure()
ax = plt.gca()
ax.quiver(0,0,xs,ys,angles='xy',scale_units='xy',scale=1, linewidth = .01)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')
plt.draw()
plt.show()

# Compute document similarity using LSA components
similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
aa = pd.DataFrame(similarity,index=all_files, columns=all_files).head(69)

# plot heatmap
plt.rcParams.update({'font.size': 3})
ax = sns.heatmap(aa.T)#,annot=aa)
plt.tight_layout()

#plt.pcolor(aa)
#plt.yticks(np.arange(0.01, len(aa.index), 1), aa.index)
#plt.xlabel('Each document of fairytale')
#plt.xticks(np.arange(1.2, len(aa.columns), 2), aa.columns)
#plt.title('Plot heatmap of ')
plt.show()
print(aa)
