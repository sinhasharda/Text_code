# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:54:32 2016

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
from sklearn.metrics.pairwise import euclidean_distances


def top3_ques(data):
    # load nltk's English stopwords as variable called 'stopwords'
    #stopwords = nltk.corpus.stopwords.words('english')
    
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
    
    
    
    top3quesList= ""
    
    articles= data['Query_Str'].values.tolist()
    articles = [x.decode('windows-1252') for x in articles]        
    totalvocab_stemmed=[]
    totalvocab_tokenized= []
    for i in articles:
        totalvocab_stemmed.extend(tokenize_and_stem(i))
        totalvocab_tokenized.extend(tokenize_only(i))
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    vocab_frame= vocab_frame.drop_duplicates(subset=['words'])
    
    if (len(articles)== 1):
        top3quesList= articles[0]
    elif(len(articles)>1 and len(articles)<4):
        for i in np.unique(articles).tolist():
            top3quesList= top3quesList + "*" + i  
    else:
       
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
        uniqueclust= np.unique(clusters)
        
        #Seeing the corresponding row with the cluster number    
        #for i in xrange(len(articles)):
        #    print ("cluster: ",clusters[i] ,"; information: ", articles[i])
        
        #Clustered Data Frame
        query_hclust= {'Query_Str': articles, 'Cluster': clusters_hclust}
        clustered_data_hclust= pd.DataFrame(query_hclust,index = [clusters_hclust], columns = ['Query_Str', 'Cluster'])
        clustered_data_hclust['Cluster'].value_counts()
        #clustered_data_hclust.to_csv('D:\mCaas\Top 3 ques\cluster_hclust_'+company+'.csv', sep=',', encoding='utf-8')
        hcluster.leaders(linkage, clusters)
        
        #Getting number of rows in each cluster and arranging in descending order   
        aggr_cluster = clustered_data_hclust.groupby(["Cluster"]).size().reset_index(name="Count")
        cluster_select=aggr_cluster.sort("Count",ascending = False)
        print cluster_select        
        
        if (len(uniqueclust) >= 3):
            top3cluster= cluster_select.head(3)
            for i in top3cluster.Cluster:
                clst_df= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == i)])
                clst_df= clst_df.reset_index(drop=True)
                if (len(clst_df)>1):
                    tfidf_clst= tfidf_vectorizer.fit_transform(clst_df.Query_Str)
                    clst_cent= cal_centroid(tfidf_clst)
                    pos= pd.DataFrame(euclidean_distances(tfidf_clst, clst_cent ),columns = ["distance"]).sort('distance').head(1).index
                    top3quesList= top3quesList +"*" + clst_df['Query_Str'].iloc[pos].values[0]
                else:
                    top3quesList= top3quesList + "*" + str(clst_df.Query_Str[0])
            
        elif(len(uniqueclust) == 2):
            top3cluster= cluster_select.head(2)
            #Adding first two Questions to list of Top3
            firstdf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[0])])
            firstdf= firstdf.reset_index(drop=True)
            if (len(firstdf)>1):
                tfidf_first= tfidf_vectorizer.fit_transform(firstdf.Query_Str)
                first_cent= cal_centroid(tfidf_first)
                first_pos= pd.DataFrame(euclidean_distances(tfidf_first, first_cent ),columns = ["distance"]).sort('distance').head(2).index.values
                for i in first_pos:
                    top3quesList= top3quesList + "*" + firstdf.Query_Str.iloc[first_pos][i]
        
            else:
                top3quesList= top3quesList + "*" + firstdf.Query_Str[0]
 
            #Adding second Question to list of Top3
            seconddf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[1])])
            seconddf= seconddf.reset_index(drop=True)
            if (len(seconddf)>1): 
                tfidf_second= tfidf_vectorizer.fit_transform(seconddf.Query_Str)
                second_cent= cal_centroid(tfidf_second)
                #second_pos= np.where(min(euclidean_distances(second_cent, tfidf_second)))
                second_pos= pd.DataFrame(euclidean_distances(tfidf_second, second_cent ),columns = ["distance"]).sort('distance').head(1).index.values
                top3quesList= top3quesList + "*" + seconddf.Query_Str[second_pos].values[0]
            else:
                top3quesList= top3quesList + "*" + seconddf.Query_Str[0]
                
        elif(len(uniqueclust) == 1):
            top3cluster= cluster_select.head(1)
            firstdf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[0])])
            firstdf= firstdf.reset_index(drop=True)
            if (len(firstdf)>1):
                tfidf_first= tfidf_vectorizer.fit_transform(firstdf.Query_Str)
            
                first_cent= cal_centroid(tfidf_first)
            
                first_pos= pd.DataFrame(euclidean_distances(tfidf_first, first_cent ),columns = ["distance"]).sort('distance').head(3).index.values
                
                for i in first_pos:
                    top3quesList= top3quesList + "*" + firstdf.Query_Str.iloc[first_pos][i]
     
            else:
                top3quesList= top3quesList + "*" + firstdf.Query_Str[0]
                    
        
    return top3quesList
        
 
