# -*- coding: utf-8 -*-
"""
Created on Wed May 15 01:49:38 2024

@author: Asus
"""
#importing lib 

import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

#declaring varibale 
text = "Word2vec is a 2technique in natural, 3language 3processing (NLP) for]4 obtaining vector representations of words. These vectors capture information about the meaning of the word based on the surrounding words. The word2vec algorithm estimates these representations by modeling text in a large corpus"

#text preprocessing 
text = re.sub(r'\d', ' ',text)
text = re.sub(r'[^\w\s]', ' ',text)
text = re.sub(r'\[[0-9]*\]',' ',text)
print(text)

# Preparing the dataset
import nltk
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

#Training the model
model = Word2Vec(sentences, min_count=1)    
print(model.wv.key_to_index)
similar_words = model.wv.most_similar('word')
print("Similar words to 'word':", similar_words)
