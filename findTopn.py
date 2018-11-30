# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:10:57 2016

@author: Sharda.sinha
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

def findTopnques(company, n):
    def cal_centroid(mat):
        cent= np.mean(mat, axis=0)
        return cent
    
    topNquesList= ""
    #pat = '/root/python-apps/topNques/Model/cluster_hclust_'+company+'.csv' 
    pat = 'D:\\mCaas\\Top 3 ques\\Trials\\Clusters\\cluster_hclust_'+company+'.csv' 
    df= pd.read_csv(pat)
    
    uniqueclust= np.unique(df.Cluster)
    
    #Getting number of rows in each cluster and arranging in descending order   
    aggr_cluster = df.groupby(["Cluster"]).size().reset_index(name="Count")
    cluster_select=aggr_cluster.sort("Count",ascending = False)
    print cluster_select  
    
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english',
                                     use_idf=True, ngram_range=(1,3), decode_error='ignore')
            
    if(len(uniqueclust) < n):
        n= len(uniqueclust)
    topNcluster= cluster_select.head(n)
    for i in topNcluster.Cluster:
        clst_df= pd.DataFrame(df.loc[(df.Cluster == i)])
        clst_df= clst_df.reset_index(drop=True)
        if (len(clst_df)>1):
            tfidf_clst= tfidf_vectorizer.fit_transform(clst_df.Query_Str)
            clst_cent= cal_centroid(tfidf_clst)
            pos= pd.DataFrame(euclidean_distances(tfidf_clst, clst_cent ),columns = ["distance"]).sort('distance').head(1).index
            topNquesList= topNquesList +"*" + clst_df['Query_Str'].iloc[pos].values[0]
        else:
            topNquesList= topNquesList + "*" + str(clst_df.Query_Str[0])
            
    return topNquesList
            
  
        
