# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:58:10 2016

@author: Sharda.sinha
"""

import hdbscan
import pandas as pd
#import datetime
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import arange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#df= pd.read_csv('D:\mCaas\Top 3 ques\Top3Data.csv')
df= pd.read_csv('D:\mCaas\TopNQues\python-apps\Input\Top3Data_updated.csv')
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




#Take input from user
company= raw_input("Enter Company Name: ")
#duration= pd.to_datetime(raw_input("Enter duration(in days): ")) #duration configurable


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
    
    
    
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
    
    plt.scatter(tfidf_matrix.T[0], tfidf_matrix.T[1], color='b', **plot_kwds)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=False, gen_min_span_tree=True)
    clusterer.fit(tfidf_matrix)
    n_clusters_ = len(set(clusterer.labels_)) - (1 if -1 in (clusterer.labels_) else 0)
    clusters_hdb= clusterer.labels_.tolist()
    labels = clusterer.labels_
    query_hdb= {'Query_Str': articles, 'Cluster': clusters_hdb}    
    clustered_data_hdb= pd.DataFrame(query_hdb ,index = [clusters_hdb], columns = ['Query_Str', 'Cluster'])
    clustered_data_hdb['Cluster'].value_counts()
    clustered_data_hdb.to_csv('D:\mCaas\Top 3 ques\cluster_dbscan_'+company+'.csv', sep=',', encoding='utf-8')
    
     #Getting number of rows in each cluster and arranging in descending order   
    aggr_cluster = clustered_data_hdb.groupby(["Cluster"]).size().reset_index(name="Count")
    cluster_select=aggr_cluster.sort("Count",ascending = False)
    print cluster_select    