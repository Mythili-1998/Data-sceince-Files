# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:00:31 2022

@author: Mythili N
"""
# Load the Data
import numpy as np
import pandas as pd 
df = pd.read_csv("D:\\data science\\Assignments\\clustering\\EastWestAirlines.csv")
df.shape
list(df)
df.head()
type(df)

# Taking varibles as x
x = df.iloc[:,0:12]
x.shape
list(x)

# Standardization
from sklearn.preprocessing import StandardScaler 
Scaler = StandardScaler()
Scaler.fit(x)
x = Scaler.transform(x)
x

# K-Means Clustering
# Visualization Plot
%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], x[:,3])
plt.show()

# intializing Kmeans
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=5)
# fitting with inputs
KM1 = KM.fit(x)
# predicting the clusters
labels = KM1.predict(x)
type(labels)
l1 = pd.DataFrame(labels)
l1[0].value_counts()

# Getting the clusters
C = KM1.cluster_centers_
C
# Total with in centroid sum of squares 
KM1.inertia_

# Another Plot with centroids
%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0],x[:,1],x[:,2],x[:,3])
ax.scatter(C[:,0],C[:,1],C[:,2],C[:,3],marker='*',c='red', s=1000)

y = pd.DataFrame(labels)

df_mew = pd.concat([pd.DataFrame(x),y],axis=1)
pd.crosstab(y[0],y[0])
y

# if we apply k =5, we could see a good variation till k=5, later on from k=4
# we dont see much change from k=4 and other above values
# Hence through k-means clustering i recommend k=5 is gives 
# best number of clusters


# Elbow Plot or Scree Plot
clust = []
for i in range(1, 5):
    KM1 = KMeans(n_clusters=i).fit(x)    
    labels = KM1.predict(x)
    clust.append(KM1.inertia_)
    
plt.plot(range(1, 5), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

######################################################################
# Hirerichal Clustering

import scipy.cluster.hierarchy as shc

# construction of dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("customer dendograms")
dend = shc.dendrogram(shc.linkage(x , method='ward'))
plt.show()

# forming a group usin clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,linkage='ward')
l2 = cluster.fit_predict(x)
l2
l3 = pd.DataFrame(l2)
l3[0].value_counts()

# I have tried with ward linkage and it looks better with other linkage
# methods and i could see same number of clusters and values are formed as
# K-means have suggested.
# we can proceed with 5 clusters


###############################################################################

# DBSCAN
# Select the model
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=7, min_samples=10)
dbscan.fit(x)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster1'])
cl
cl['cluster1'].value_counts()
cl

clustered = pd.concat([pd.DataFrame(x),cl],axis=1)
clustered.head()

clustered['cluster1'].value_counts()

noisedata = clustered[clustered['cluster1']==-1]
noisedata

finaldata = clustered[clustered['cluster1']==0]
finaldata


clustered.mean()
finaldata.mean()

# after applying DBSCAN with many min_samples and epsilon values 
# we could see three outlier is formed from whole data 
# hence we recommend we can remove that values from the data


