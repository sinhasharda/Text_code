# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 15:19:39 2016

@author: Sharda.sinha
"""

import pandas as pd
#import datetime
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import arange
#from scipy import spatial
#import fastcluster
import scipy.cluster.hierarchy as hcluster
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
df= pd.read_csv('D:\mCaas\Top 3 ques\Trials\Top3Data_updated.csv')
#df= pd.read_csv('D:\mCaas\Top 3 ques\Top 3_2nd_Dataset.csv')
quesdf= df. dropna(subset=['Query_Str'])
#quesdf["DateTime"]= pd.to_datetime(quesdf["DateTime"])
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
print (stopwords)


stemmer = SnowballStemmer("english", ignore_stopwords=True) # stems the word example - running to run, ignoring stopping words like having etc
print (stemmer.stem('running'))

#here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def cal_centroid(mat):
        cent= np.mean(mat, axis=0)
        return cent


#Take input from user
company= raw_input("Enter Company Name: ")
#duration= pd.to_datetime(raw_input("Enter duration(in days): ")) #duration configurable


if (any(quesdf.Company == company)):
    print "Your company is there"
    top3quesList= ""
     
    #Getting the corresponding data for the entered company and duration
    #data = pd.DataFrame(quesdf.loc[((quesdf.Company == company) & ((datetime.datetime.now()- quesdf.DateTime)== duration)),'Query_Str'], columns= ['Query_Str'])
    
    data = pd.DataFrame(quesdf.loc[(quesdf.Company == company) ,'Query_Str'], columns= ['Query_Str'])
    articles= data['Query_Str'].values.tolist()
    articles = [x.decode('windows-1252') for x in articles]
    if (len(articles)== 1):
        top3quesList= articles
    elif(len(articles)>1 and len(articles)<4):
        for i in articles:
            top3quesList= top3quesList + "*" + i
        
    
    totalvocab_stemmed=[]
    totalvocab_tokenized= []
    for i in articles:
        totalvocab_stemmed.extend(tokenize_and_stem(i))
        totalvocab_tokenized.extend(tokenize_only(i))
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    vocab_frame= vocab_frame.drop_duplicates(subset=['words'])
    
    #Tfidf matrix
    #tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
     #                            min_df=0.01, stop_words='english',
      #                           use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english',
                                 use_idf=True, ngram_range=(1,3),decode_error='ignore')
    
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
    
    