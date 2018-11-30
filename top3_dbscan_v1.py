# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:03:43 2016

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

df= pd.read_csv('D:\mCaas\Top 3 ques\Top3Data.csv')
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

def top3_ques(quesdf,company):
    if (any(quesdf.Company == company)):
        print "Your company is there"
    
     
        #Getting the corresponding data for the entered company and duration
        #data = pd.DataFrame(quesdf.loc[((quesdf.Company == company) & ((datetime.datetime.now()- quesdf.DateTime)== duration)),'Query_Str'], columns= ['Query_Str'])
        
        data = pd.DataFrame(quesdf.loc[(quesdf.Company == company) ,'Query_Str'], columns= ['Query_Str'])
        articles= data['Query_Str'].values.tolist()
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
                                     use_idf=True, ngram_range=(1,3))
        
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
        
        #DBSCAN Clustering
        
        db = DBSCAN(eps=0.3, min_samples=1).fit(tfidf_matrix)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True    
        clusters_db= db.labels_.tolist()
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in (labels) else 0)
        
        query_db= {'Query_Str': articles, 'Cluster': clusters_db}    
        clustered_data_db= pd.DataFrame(query_db ,index = [clusters_db], columns = ['Query_Str', 'Cluster'])
        clustered_data_db['Cluster'].value_counts()
        clustered_data_db.to_csv('D:\mCaas\Top 3 ques\cluster_dbscan_'+company+'.csv', sep=',', encoding='utf-8')
    