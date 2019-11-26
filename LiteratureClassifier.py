import spacy
import urllib
import os
import tensorflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
pca = PCA(n_components=2)


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
	counter+=1
	if counter == 12:
		break
	with open(filename, encoding="utf8") as file:
    	txt=file.read()
    	documents.append(txt)

for filename in glob.iglob("DataScienceRawData/mystery/*.txt"):
	counter+=1
	if counter == 12:
		break
	with open(filename, encoding="utf8") as file:
    	txt=file.read()
    	documents.append(txt)

#data fit in 


y_train=np.arrary([np.zeros(6),np.ones(6)]).flatten()
#remove the project info and empty lines
def removeprointo(txt):
    txt=txt[txt.find('***')+2:]  
    txt=txt[txt.find('***')+3:]
    text = os.linesep.join([s for s in txt.splitlines() if s])
    return text


X_t=np.zeros((24,3839616))

def nlp2vector(str1):
	
	doc4	 = nlp(str1.replace("\n", " "))
	vectors = np.vstack([word.vector for word in doc4 if word.has_vector])
	return vectors[0:9999].flatten()

for i in range(24):

	
	
	X_t[i]=nlp2vector(documents[i])

for token in doc4:
	print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,)
# token.lemma_ is the word lemma 
#for instance : 'has' and 'having' and 'had' lemma are all 'have'

#word to vectors



#compress with PCA
pca = PCA(n_components=2)
# vecs_transformed = pca.fit_transform(vectors)


#test dataset

with open("DataScienceRawData/fairytale/A Tangled Tale.txt") as file:
    txt5=file.read()
with open("DataScienceRawData/mystery/A Duet, with an Occasional Chorus.txt") as file:
    txt9=file.read()

t5=removeprointo(txt5)
t9=removeprointo(txt9)
v7=nlp2vector(t5)
v9=nlp2vector(v9)
test=np.array([v7,v9])

# Naive Bayse 
gnb.fit(X_t, y_train)

y_pred = gnb.predict(test) 
print(y_pred)


#neural network


