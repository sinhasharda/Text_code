# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:48:05 2016

@author: Sharda.sinha
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import ward, dendrogram
from scipy import spatial
ds = pd.read_csv('D:\mCaas\Top 3 ques\Top3Data.csv')

#Data Pre-processing
#Remove rows with NA in Query_Str
ds= ds.dropna(subset=['Query_Str'])
#Remove rows with '.' in Query_Str
ds=ds[ds['Query_Str'].str.startswith('.')== False]
ds=ds[ds['Query_Str'].str.startswith('<')== False]


# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
print (stopwords)


# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english", ignore_stopwords=True) # stems the word example - running to run, ignoring stopping words like having etc
print (stemmer.stem('eating'))

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
    

#Enter the company name
ask = raw_input("Enter the company:")

#Checking the condition if company is present in the data
if (any(ds.Company == ask)):
    print "Your company is there"
    #Getting the corresponding data for the entered company
    data =pd.DataFrame(ds.loc[ds.Company == ask,'Query_Str'])
    
    #Converting to data frame and then to list
    data = pd.DataFrame({"Id": data.index, "Query_Str":data.Query_Str})
    print data    
    articles=data['Query_Str'].values.tolist()
    #Creating empty list to store the values 
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in articles:
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'articles', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    print (totalvocab_tokenized)
    print (totalvocab_stemmed)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    print (vocab_frame)

    #Tfidf matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(articles) #fit the vectorizer to articles
    print(tfidf_matrix.shape)
    
    terms = tfidf_vectorizer.get_feature_names()
    idf = tfidf_vectorizer.idf_
    print(terms)
    print dict(zip(terms,idf))
    
    
    
    
    dist = 1 - cosine_similarity(tfidf_matrix)    
    
    dist2= spatial.distance.pdist(tfidf_matrix.toarray())
    

    linkage_matrix = ward(dist2) #define the linkage_matrix using ward clustering pre-computed distances
    
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="left", labels= data["Id"].values.tolist());
   
    plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

    plt.tight_layout() #show plot with tight layout
    plt.close()
    #clustering
    thresh = 0.5* dist.max()
    clusters = hcluster.fclusterdata(tfidf_matrix, thresh, criterion="distance", method="ward")
  
    clust= hcluster.fcluster()
    cluster2 = hcluster.fcluster(linkage_matrix, thresh, criterion="distance")
    
#############################################################

def plot_cluster(cluster, sample_matrix):
    '''Input:  "cluster", which is an object from DBSCAN, 
       e.g. dbscan_object = DBSCAN(3.0,4)
"sample_matrix" which is a data matrix:  
X = [
    [0,5,1,2],
    [0,4,1,3],
    [0,5,1,3],
    [0,5,0,2],
    [5,5,5,5],
    ]
        Output: Plots the clusters nicely.    
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    f = lambda row: [float(x) for x in row]

    sample_matrix = map(f,sample_matrix)
    print sample_matrix
    sample_matrix = StandardScaler().fit_transform(sample_matrix)

    core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
    core_samples_mask[cluster.core_sample_indices_] = True
    labels = cluster.labels_

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)  # generator comprehension 
        # X is your data matrix
        X = np.array(sample_matrix)

        xy = X[class_member_mask & core_samples_mask]

        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.ylim([0,10]) 
    plt.xlim([0,10])    
#        plt.title('Estimated number of clusters: %d' % n_clusters_)


    
    
    
    
    
    
    
    