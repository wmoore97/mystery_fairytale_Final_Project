import spacy
#import urllib
import os
import glob
import tensorflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB 
import numpy as np
gnb = GaussianNB() 
#pca = PCA(n_components=2)


"""
@author:Yulong

"""

nlp = spacy.load("en_core_web_sm")

# deal with raw data directly



#read online

# url1="https://raw.githubusercontent.com/DigiUGA/Gutenberg_Text/master/Wilde%2C%20Oscar/A%20Woman%20of%20No%20Importance.txt"
# text = urllib.request.urlopen(url1).read().decode("utf-8") 


documents = []
counter=0

#or read locally
for filename in glob.iglob("DataScienceRawData/fairytale/*.txt"):
	
	if counter == 12:
		break
	with open(filename, encoding="utf8") as file:
		txt=file.read()
		documents.append(txt)
	counter+=1	
counter=0
for filename in glob.iglob("DataScienceRawData/mystery/*.txt"):
	
	if counter == 12:
		break
	with open(filename, encoding="utf8") as file:
		txt=file.read()
		documents.append(txt)
	counter+=1	
#data fit in 


y_train=np.array([np.zeros(12),np.ones(12)]).flatten()


#remove the project info and empty lines
def removeprointo(txt):
    txt=txt[txt.find('***')+2:]  
    txt=txt[txt.find('***')+3:]
    text = os.linesep.join([s for s in txt.splitlines() if s])
    return text


X_t=np.zeros((24,3839616))
#word to vectors

def nlp2vector(str1):
	
	doc4	 = nlp(str1.replace("\n", " "))
	vectors = np.vstack([word.vector for word in doc4 if word.has_vector])
	return vectors[0:9999].flatten()



#for token in doc4:
#	print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,)
# token.lemma_ is the word lemma 
#for instance : 'has' and 'having' and 'had' lemma are all 'have'


j=0
for i in documents:
	tep=removeprointo(i)
	tem=nlp2vector(tep)
	
	X_t[j]=tem
	print("currently processing traing one by one :"+str(j))
	j+=1


#compress with PCA
#pca = PCA(n_components=2)
# vecs_transformed = pca.fit_transform(vectors)


#test dataset 
print("goes to test data ")
with open("DataScienceRawData/fairytale/Three Sunsets and Other Poems.txt") as file:
    txt5=file.read()

with open("DataScienceRawData/mystery/Rodney Stone.txt") as file:
    txt9=file.read()

with open("DataScienceRawData/fairytale/Three Sunsets and Other Poems.txt") as file:
    txt10=file.read()
with open("DataScienceRawData/fairytale/Wonderful Stories for Children.txt") as file:
    txt11=file.read()
with open("DataScienceRawData/mystery/Victorian Short Stories of Troubled Marriages.txt") as file:
    txt19=file.read()

with open("DataScienceRawData/mystery/Uncle Bernac_ A Memory of the Empire.txt") as file:
    txt29=file.read()   


t5=removeprointo(txt5)
t9=removeprointo(txt9)
t10=removeprointo(txt10)
t11=removeprointo(txt11)
t19=removeprointo(txt19)
t29=removeprointo(txt29)
v7=nlp2vector(t5)
v9=nlp2vector(t9)
v10=nlp2vector(t10)
v11=nlp2vector(t11)
v19=nlp2vector(t19)
v29=nlp2vector(t29)
test=np.array([v7,v9,v10,v11,v19,v29])
print("traing")
# Naive Bayse 
gnb.fit(X_t, y_train)
print("training done")
y_pred = gnb.predict(test) 
print(y_pred)


#neural network


