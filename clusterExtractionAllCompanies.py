# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:59:51 2016

@author: Sharda.sinha
"""
import pandas as pd
import sys
import numpy as np
#sys.path.insert(0, '/root/python-apps/topNques/Model')
#import topnQuesClusters_v1
sys.path.insert(0, 'D:\\mCaas\\Top 3 ques\\Trials')
import topnQuesClusters_v1


def extractClusters_allcompanies():
    #Reading the respective tenant folder
    #pat = '/root/python-apps/Input/Top3Data_updated.csv'       
    pat= 'D:\mCaas\Top 3 ques\Trials\Top3Data_updated.csv'
    df= pd.read_csv(pat)
    
    quesdf= df. dropna(subset=['Query_Str'])
    #quesdf["DateTime"]= pd.to_datetime(quesdf["DateTime"])
    
    companyList= np.unique(quesdf['Company']).tolist()

    for company in companyList:
        #Getting the corresponding data for the entered company and duration
        #data = pd.DataFrame(quesdf.loc[((quesdf.Company == company) & ((datetime.datetime.now()- quesdf.DateTime)== duration)),'Query_Str'], columns= ['Query_Str'])      
        data = pd.DataFrame(quesdf.loc[(quesdf.Company == company) ,'Query_Str'], columns= ['Query_Str']) 
        clustersdf =topnQuesClusters_v1.quesClusters(data)
        
        clustersdf.to_csv('D:\\mCaas\\Top 3 ques\\Trials\\Clusters\\cluster_hclust_'+company+'.csv', sep=',', encoding='utf-8')
        #clustersdf.to_csv('/root/python-apps/topNques/Model/cluster_hclust_'+company+'.csv', sep=',', encoding='utf-8')

extractClusters_allcompanies()   