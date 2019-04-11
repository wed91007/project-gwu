# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 23:24:17 2019

@author: Administrator
"""

import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score,v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score 
import time

np.random.seed(123)
Data = pd.read_csv('input/train.csv')
Data.sample(5)

print('Shape of the data set:'+ str(Data.shape))

#save labels as string
Labels = Data['activity']
Data = Data.drop(['rn','activity'],axis=1)#drop delete row,axis=1 to delete column
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)
print('Activity labels:'+str(Labels_keys))

#check for missing values
Temp = pd.DataFrame(Data.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])) )

scaler = StandardScaler()
Data = scaler.fit_transform(Data)

#check for optimal k value
ks=range(1,10)
inertias=[]
start = time.clock()
for k in ks:
    model=KMeans(n_clusters=k)
    model.fit(Data)
    inertias.append(model.inertia_)
    
plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()

def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels 
    and clustering performance parameters.
    
    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels
    
    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(k_means.inertia_,
      homogeneity_score(true_labels, y_clust),
      completeness_score(true_labels, y_clust),
      v_measure_score(true_labels, y_clust),
      adjusted_rand_score(true_labels, y_clust),
      adjusted_mutual_info_score(true_labels, y_clust),
      silhouette_score(data_frame, y_clust, metric='euclidean')))
    

k_means(n_clust=2, data_frame=Data, true_labels=Labels)
elapsed = (time.clock()-start)
print("Time used:",elapsed)

#PCA
pca = PCA(random_state=123)
pca.fit(Data)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()