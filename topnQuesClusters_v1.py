# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:50:58 2016

@author: Sharda.sinha
"""

import pandas as pd
#import datetime
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.cluster.hierarchy as hcluster
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def quesClusters(data):
    # load nltk's English stopwords as variable called 'stopwords'
    #stopwords = nltk.corpus.stopwords.words('english')
    
    print "got data"
    stemmer = SnowballStemmer("english", ignore_stopwords=True) # stems the word example - running to run, ignoring stopping words like having etc
    
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
   
    articles= data['Query_Str'].values.tolist()
    articles = [x.decode('windows-1252') for x in articles] 
    print "found articles"       
    totalvocab_stemmed=[]
    totalvocab_tokenized= []
    for i in articles:
        totalvocab_stemmed.extend(tokenize_and_stem(i))
        totalvocab_tokenized.extend(tokenize_only(i))
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    vocab_frame= vocab_frame.drop_duplicates(subset=['words'])
    clustered_data_hclust= pd.DataFrame(columns = ['Query_Str', 'Cluster'])
   
    if len(articles)<=20:
         clustered_data_hclust['Query_Str']=np.unique(articles).tolist()
         for i,r in clustered_data_hclust.iterrows():
             r['Cluster']= 1
         print "less than 20"
         print clustered_data_hclust     
    else:
        articles= np.unique(articles).tolist()
        #Tfidf matrix
        #tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
        #                            min_df=0.01, stop_words='english',
        #                           use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
        
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english',
                                     use_idf=True, ngram_range=(1,3), decode_error='ignore')
        
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
        
        
        distance= hcluster.distance.pdist(tfidf_matrix.toarray(),'euclidean')
        #distance = 1- cosine_similarity(tfidf_matrix)    
        linkage= hcluster.linkage(distance, method= 'complete')
      
        thresh = 0.99* distance.max()
        
        clusters = hcluster.fcluster(linkage, thresh, criterion="distance")  
        clusters_hclust= clusters.tolist()
        
        #Seeing the corresponding row with the cluster number    
        #for i in xrange(len(articles)):
        #    print ("cluster: ",clusters[i] ,"; information: ", articles[i])
        
        #Clustered Data Frame
        query_hclust= {'Query_Str': articles, 'Cluster': clusters_hclust}
        print query_hclust
        clustered_data_hclust= pd.DataFrame(query_hclust, index = [clusters_hclust], columns = ['Query_Str', 'Cluster'])
        print "df after append"
        print clustered_data_hclust
        clustered_data_hclust['Cluster'].value_counts()
        print clustered_data_hclust
        #clustered_data_hclust.to_csv('D:\mCaas\Top 3 ques\cluster_hclust_'+company+'.csv', sep=',', encoding='utf-8')
    return clustered_data_hclust