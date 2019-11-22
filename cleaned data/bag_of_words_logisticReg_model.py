# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:00:22 2019

@author: Moore
"""

import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
path = "."
documents = []

for filename in glob.iglob(path+"/fairytale/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        documents.append(lines)
        f.close()

for filename in glob.iglob(path+"/updated_cleaned_mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        documents.append(lines)
        f.close()

vectorizer = CountVectorizer(max_features = 1000, min_df=5,max_df=0.7)
X = vectorizer.fit_transform(documents).toarray()
m = vectorizer.get_feature_names()

y = []

k = 0
while k < 84:
    if k < 15:
        y.append(0)
        k += 1
    else:
        y.append(1)
        k+=1

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=0)

logModel = LogisticRegression()
logModel.fit(x_train,y_train)
predictions = logModel.predict(x_test)
score = logModel.score(x_test,y_test)
print(score)