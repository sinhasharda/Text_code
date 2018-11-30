# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 12:54:40 2016

@author: Sharda.sinha
"""
import pandas as pd
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import ward, dendrogram
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv('D:\mCaas\Top 3 ques\sflower.csv')

 
distance = hcluster.distance.pdist(ds)

#linkage_matrix = ward(distance)
linkage_matrix= hcluster.linkage(distance, method="average")
thresh = 0.5* distance.max()

cluster2 = hcluster.fcluster(linkage_matrix, thresh, criterion="distance")
u = np.unique(cluster2)
plt.figure(figsize=(5, 5))
plt.imshow(ds, cmap=plt.cm.gray)
for l in range(u):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(u)), ])
plt.xticks(())
plt.yticks(())
plt.show()