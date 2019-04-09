# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 15:57:17 2019

@author: Administrator
"""


#get the images and labels mnist
#-*- coding: utf-8 -*-
 
 
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#from skimage import exposure
import numpy as np
#import imutils
import cv2
import matplotlib.pyplot as plt
# load the MNIST digits dataset
dataset = pd.read_csv('F:/Python/mnist/train.csv')


Data = dataset.iloc[:,1:]
Labels = dataset.iloc[:,0]
#
#
#
##print (mnist.data)
#
## Training and testing split,
## 75% for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(Data, Labels, test_size=0.25)
#
## take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)

# Feature scaling
sc_X = StandardScaler()
trainData = sc_X.fit_transform(trainData)
testData = sc_X.transform(testData)
valData = sc_X.transform(valData)

#nor_x = MinMaxScaler()
#trainData = nor_x.fit_transform(trainData)
#testData = nor_x.transform(testData)
#valData = nor_x.transform(valData)
#bi_X = Binarizer(threshold=0.4,)
#trainData = bi_X.fit_transform(trainData)
#testData = bi_X.transform(testData)
#valData = bi_X.transform(valData)
# Checking sizes of each data split
#print("training data points: {}".format(len(trainLabels)))
#print("validation data points: {}".format(len(valLabels)))
#print("testing data points: {}".format(len(testLabels)))

for i in np.random.randint(0, high=len(Data), size=(5,)):
         # grab the image and classify it
         image = Data[i]
         label = Labels[i]
         # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
         # then resize it to 32 x 32 pixels so we can see it better
##         image = image.reshape((64, 64))
##         image = exposure.rescale_intensity(image, out_range=(0, 255))
##         image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
         
         # show the prediction
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((28,28))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(label,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print("The digit is : {}".format(label))
         #cv2.imshow("image", image)
         plt.show()
         cv2.waitKey(0)