# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 16:18:54 2016

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
    
    #Hierarchical clustering
#    distance = spatial.distance.pdist(tfidf_matrix)
#    linkage = fastcluster.linkage(distance,method="complete")


    distance= hcluster.distance.pdist(tfidf_matrix.todense(),'euclidean')
    print "distance"
    print distance
    #distance = 1- cosine_similarity(tfidf_matrix)    
    linkage= hcluster.linkage(distance, method= 'complete')
    print"linkage"
    print linkage
    #linkage= hcluster.ward(distance)
    P =hcluster.dendrogram(linkage)

    

    

    for i in arange(distance.min(), distance.max(), 0.01):
        thresh = i
        clusters = hcluster.fcluster(linkage, thresh, criterion="distance")  
        clusters_hclust= clusters.tolist()
        uniqueclust= np.unique(clusters)
        print (" at i =" ,i , "no. of clusters=" , len(uniqueclust) )


    
    #plt.savefig('D:\mCaas\Top 3 ques\plot_dendrogram.png')
    #thresh= 0.8
    thresh = 0.99*distance.max()
    
    clusters = hcluster.fcluster(linkage, thresh, criterion="distance")  
    clusters_hclust= clusters.tolist()
    uniqueclust= np.unique(clusters)
    
    #hcluster.leaders(linkage, clusters)
    
    #Seeing the corresponding row with the cluster number    
    for i in xrange(len(articles)):
        print ("cluster: ",clusters[i] ,"; information: ", articles[i])
  
   
    
    #Clustered Data Frame
    query_hclust= {'Query_Str': articles, 'Cluster': clusters_hclust}
    clustered_data_hclust= pd.DataFrame(query_hclust,index = [clusters_hclust], columns = ['Query_Str', 'Cluster'])
    
    
    
    clustered_data_hclust['Cluster'].value_counts()
   
    
    clustered_data_hclust.to_csv('D:\mCaas\Top 3 ques\cluster_hclust_'+company+'.csv', sep=',', encoding='utf-8')
    
    
    
    
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
                #print top3quesList
    elif(len(uniqueclust) == 2):
        top3cluster= cluster_select.head(2)
        #Adding first two Questions to list of Top3
        
        firstdf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[0])])
        firstdf= firstdf.reset_index(drop=True)
        if (len(firstdf)>1):
            tfidf_first= tfidf_vectorizer.fit_transform(firstdf.Query_Str)
        
            first_cent= cal_centroid(tfidf_first)
        
            #first_two_pos= np.where(min(euclidean_distances(first_cent, tfidf_first)))
            first_pos= pd.DataFrame(euclidean_distances(tfidf_first, first_cent ),columns = ["distance"]).sort('distance').head(2).index.values
            for i in first_pos:
                top3quesList= top3quesList + "*" + firstdf.Query_Str.iloc[first_pos][i]
            
        else:
            #top3quesList.append(firstdf.Query_Str[0])
            top3quesList= top3quesList + "*" + firstdf.Query_Str[0]
 
        #Adding second Question to list of Top3
        seconddf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[1])])
        seconddf= seconddf.reset_index(drop=True)
        if (len(seconddf)>1): 
            tfidf_second= tfidf_vectorizer.fit_transform(seconddf.Query_Str)
        
            second_cent= cal_centroid(tfidf_second)
        
            second_pos= np.where(min(euclidean_distances(second_cent, tfidf_second)))
        
            #top3quesList.append(seconddf.Query_Str[second_pos[0][0]])
            top3quesList= top3quesList + "*" + seconddf.Query_Str[second_pos[0][0]]
        else:
            #top3quesList.append(seconddf.Query_Str[0])
            top3quesList= top3quesList + "*" + seconddf.Query_Str[0]
            
    elif(len(uniqueclust) == 1):
        top3cluster= cluster_select.head(1)
        firstdf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[0])])
        firstdf= firstdf.reset_index(drop=True)
        if (len(firstdf)>1):
            tfidf_first= tfidf_vectorizer.fit_transform(firstdf.Query_Str)
        
            first_cent= cal_centroid(tfidf_first)
        
            #first_two_pos= np.where(min(euclidean_distances(first_cent, tfidf_first)))
            first_pos= pd.DataFrame(euclidean_distances(tfidf_first, first_cent ),columns = ["distance"]).sort('distance').head(3).index.values
            #top3quesList.append(firstdf.Query_Str[first_pos[0][0]])
            for i in first_pos:
                top3quesList= top3quesList + "*" + firstdf.Query_Str.iloc[first_pos][i]
            
            
        else:
            #top3quesList.append(firstdf.Query_Str[0])
            top3quesList= top3quesList + "*" + firstdf.Query_Str[0]

#            

#        #Adding third Question to list of Top3
#        thirddf= pd.DataFrame(clustered_data_hclust.loc[(clustered_data_hclust.Cluster == top3cluster.Cluster.iloc[2])])
#        thirddf= thirddf.reset_index(drop=True)
#        if (len(thirddf)>1): 
#            tfidf_third= tfidf_vectorizer.fit_transform(thirddf.Query_Str)
#            
#            third_cent= cal_centroid(tfidf_third)
#            
#            third_pos= np.where(min(euclidean_distances(third_cent, tfidf_third)))
#            
#            #top3quesList.append(thirddf.Query_Str[third_pos[0][0]])
#            top3quesList= top3quesList + "*" + thirddf.Query_Str[third_pos[0][0]]
#        else:
#            #top3quesList.append(thirddf.Query_Str[0])
#            top3quesList= top3quesList + "*" + thirddf.Query_Str[0]
 
#DBSCAN Clustering
    
    dist= hcluster.distance.pdist(tfidf_matrix.toarray(),'euclidean')
    
    
    for i in arange(dist.min(), dist.max(), 0.01):
        db = DBSCAN(eps=0.65 *dist.max(), min_samples=1).fit(tfidf_matrix)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in (db.labels_) else 0)
        print ("no of clusters: ", n_clusters_)
    
    db = DBSCAN(eps=0.65 *dist.max(), min_samples=1).fit(tfidf_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True    
    clusters_db= db.labels_.tolist()
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in (labels) else 0)
    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels== k)

        xy = tfidf_matrix[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0].toarray(), xy[:, 1].toarray(), 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

        xy = tfidf_matrix[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0].toarray(), xy[:, 1].toarray(), 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()    

query_db= {'Query_Str': articles, 'Cluster': clusters_db}    
clustered_data_db= pd.DataFrame(query_db ,index = [clusters_db], columns = ['Query_Str', 'Cluster'])
clustered_data_db['Cluster'].value_counts()            
clustered_data_db.to_csv('D:\mCaas\Top 3 ques\cluster_dbscan_'+company+'.csv', sep=',', encoding='utf-8')    
#plot clusters    
    
fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = hcluster.dendrogram(linkage_matrix, orientation="right", labels=titles);

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()   
   
###### plot the clusters
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot





