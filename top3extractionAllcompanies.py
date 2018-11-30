# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 00:35:31 2016

@author: Sharda.sinha
"""
import pandas as pd
import numpy as np
import sys
#sys.path.insert(0, '/root/python-apps/top3ques/Model')
#import top3Ques_v1
sys.path.insert(0, 'D:\mCaas\Top 3 ques\Trials')
import top3Ques_v1

def extractTop3_allcompanies():
    #Reading the respective tenant folder
    #pat = '/root/python-apps/Input/Top3Data_updated.csv'       
    pat= 'D:\mCaas\Top 3 ques\Trials\Top3Data_updated.csv'
    df= pd.read_csv(pat)
    
    quesdf= df. dropna(subset=['Query_Str'])
    #quesdf["DateTime"]= pd.to_datetime(quesdf["DateTime"])
    
    
    top3_allcompanies= pd.DataFrame(columns = ['Company', 'top3ques'])
    top3_allcompanies['Company']= np.unique(quesdf['Company']).tolist()
 
    for company in top3_allcompanies['Company']:
        #company= 'ABM Industries'  
        #Getting the corresponding data for the entered company and duration
        #data = pd.DataFrame(quesdf.loc[((quesdf.Company == company) & ((datetime.datetime.now()- quesdf.DateTime)== duration)),'Query_Str'], columns= ['Query_Str'])      
        data = pd.DataFrame(quesdf.loc[(quesdf.Company == company) ,'Query_Str'], columns= ['Query_Str']) 
        top3_allcompanies.top3ques[top3_allcompanies.Company == company] =top3Ques_v1.top3_ques(data)
        
    
    top3_allcompanies.to_csv('D:\\mCaas\\Top 3 ques\\top3_allcompanies.csv', sep=',', encoding='utf-8')
    #top3_allcompanies.to_csv('/root/python-apps/top3ques/Model/top3_allcompanies.csv', sep=',', encoding='utf-8')
extractTop3_allcompanies()