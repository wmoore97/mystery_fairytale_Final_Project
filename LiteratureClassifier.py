import spacy
import urllib
import os
import tensorflow
import pandas as pd


"""
@author:Yulong

"""

nlp = spacy.load("en_core_web_sm")

# deal with raw data directly



#read online

# url1="https://raw.githubusercontent.com/DigiUGA/Gutenberg_Text/master/Wilde%2C%20Oscar/A%20Woman%20of%20No%20Importance.txt"
# text = urllib.request.urlopen(url1).read().decode("utf-8") 



#or read locally
with open("DataScienceRawData/fairytale/Alice's Adventures in Wonderland.txt") as file:
    txt=file.read()

txt=txt[txt.find('***')+2:]
txt=txt[txt.find('***')+3:] # remove the project info.

text = os.linesep.join([s for s in txt.splitlines() if s]) #remove all empty lines




#txt=open(*.txt)

doc4 = nlp(text.replace("\n", " "))

for token in doc4:
	print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,)
# token.lemma_ is the word lemma 
#for instance : 'has' and 'having' and 'had' lemma are all 'have'

#word to vectors

vectors = np.vstack([word.vector for word in doc4 if word.has_vector])

