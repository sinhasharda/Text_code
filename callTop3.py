# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:04:48 2016

@author: Sharda.sinha
"""
import pandas as pd
def callTop3ques(company):
    #pat = '/root/python-apps/top3ques/Model/top3_allcompanies.csv' 
    pat = 'D:\\mCaas\\Top 3 ques\\top3_allcompanies.csv' 
    df= pd.read_csv(pat)
    top3questions= "Your company is not found!!"
    #print company    
    for i,r in df.iterrows():
        if r['Company'] == company:
            top3questions = df.top3ques[i]
        
    return top3questions
    
result= callTop3ques("SunTrust")