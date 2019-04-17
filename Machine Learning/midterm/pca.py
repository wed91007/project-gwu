# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:39:26 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
print "step 1: generating data..."
data=[[0 for i in range(2)]for j in range(20)]
for i in range(1,11):
    data[2*i-1]=[i,i+1]
    data[2*i-2]=[i,i]
print data

data=np.mat(data)
#compute mean,row is sample number, column is feature
def meanX(dataX):
    return np.mean(dataX,axis=0)
    
def pca(XMat,k):
    average = meanX(XMat)
    m,n = np.shape(XMat)
    data_adjust = []
    avgs= np.tile(average,(m,1))#average mat
    data_adjust = XMat - avgs
#    print data_adjust.T
    covX = np.cov(data_adjust.T)
    print "covariance matrix for X is:"
    print covX
    featValue, featVec = np.linalg.eig(covX)#compute eigen value for covariance
    print "Two principal component is:"
    print featVec
    index = np.argsort(-featValue)#sort feature from high to low
    finalData = []
    if k>n:
        print "K must lower than feature number"
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])#eigen value is column so we transpose
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec)+ average
    return finalData,reconData

def plotBestFit(data1, data2):    
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0]) 
        axis_y2.append(dataArr2[i,1])                 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()  

np.set_printoptions(precision=3)
print "step 2: running PCA..."    
finaldata,reconmat = pca(data,2)
print "final data is:"
print finaldata.T
print "recon mat is:\n",reconmat.T
print "step 3: plot..."
plotBestFit(finaldata,reconmat)