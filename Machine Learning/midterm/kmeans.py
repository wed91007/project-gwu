# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:30:46 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
 
 
# calculate Euclidean distance
def euclDistance(vector1, vector2):
    a=np.sqrt(sum(np.power(vector2 - vector1, 2)))
#    print "length of a:",len(a)
#    print a[0,1]
    r=0
    for i in range(len(a)+1):
        r +=np.power(a[0,i],2)
    r = np.sqrt(r)
    return r 
 
# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    arr=[0,3,5]
    for i in range(k):
#        index = int(np.random.uniform(0, numSamples))
        index=arr[i]
        centroids[i, :] = dataSet[index, :]
        
    return centroids

def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True
 
	## step 1: init centroids
    centroids = initCentroids(dataSet, k)
    print "Centroids are:\n",centroids
    i=0
    while clusterChanged:
        clusterChanged = False
		## for each sample
        for i in xrange(numSamples):
            minDist  = 100000.0
            minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
			
			## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
 
		## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)
#        break
    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	if dim != 2:
		print "Sorry! I can not draw because the dimension of your data is not 2!"
		return 1
 
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print "Sorry! Your k is too large! please contact Zouxy"
		return 1
 
	# draw all samples
	for i in xrange(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
 
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
 
	plt.show()


print "step 1:load data..."
dataSet = np.zeros((7,2),dtype=np.int)
dataSet = [[2,10],[2,5],[8,4],[5,8],[6,4],[1,2],[4,9]]
print("step 2: clustering...")
dataSet = np.mat(dataSet)
#a=np.linalg.norm(dataSet[1]-dataSet[2])
#b=euclDistance(dataSet[1],dataSet[2])
c,ca=kmeans(dataSet,3)
print "step 3: show the result..."
#print "First round complete, centroids are:\n",c
print "The final result is:centroid:\n",c,"\n"
print "assignment:\n",ca
showCluster(dataSet,3,c,ca)

#centroid=initCentroids(dataSet,3)

