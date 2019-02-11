# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:16:03 2019

@author: Administrator
"""

# K-Nearest Neighbor Classification

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
#from sklearn import datasets
import pandas as pd
#from skimage import exposure
import numpy as np
#import imutils
import cv2
import matplotlib.pyplot as plt
import time
# load the MNIST digits dataset
#mnist = datasets.load_digits()
start = time.clock()
dataset = pd.read_csv('F:/Python/mnist/train.csv')


Data = dataset.iloc[:,1:]
Labels = dataset.iloc[:,0]



#print (mnist.data)

# Training and testing split,
# 75% for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(Data, Labels, test_size=0.25)

# take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)

# Feature scaling
#sc_X = StandardScaler()
#trainData = sc_X.fit_transform(trainData)
#testData = sc_X.transform(testData)
#valData = sc_X.transform(valData)
##
#nor_x = MinMaxScaler()
#trainData = nor_x.fit_transform(trainData)
#testData = nor_x.transform(testData)
#valData = nor_x.transform(valData)
#bi_X = Binarizer(threshold=0.4,)
#trainData = bi_X.fit_transform(trainData)
#testData = bi_X.transform(testData)
#valData = bi_X.transform(valData)
# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over kVals
for k in xrange(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))


# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

# Predict labels for the test set
predictions = model.predict(testData)

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

print("Confusion matrix")
print(confusion_matrix(testLabels,predictions))
# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
#for i in np.random.randint(0, high=len(testLabels), size=(5,)):
#    # np.random.randint(low, high=None, size=None, dtype='l')
#    image = testData[i]
#    prediction = model.predict(image)[0]
#
#    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
#    # then resize it to 32 x 32 pixels for better visualization
#    image = image.reshape((8, 8)).astype("uint8")
#    image = exposure.rescale_intensity(image, out_range=(0, 255))
#    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
#
#    # show the prediction
#    print("I think that digit is: {}".format(prediction))
#    cv2.imshow("Image", image)
#    cv2.waitKey(0) # press enter to view each one!
elapsed = (time.clock()-start)
print("Time used:",elapsed)

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
         # grab the image and classify it
         image = testData[i]
         prediction = model.predict([image])[0]
         # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
         # then resize it to 32 x 32 pixels so we can see it better
##         image = image.reshape((64, 64))
##         image = exposure.rescale_intensity(image, out_range=(0, 255))
##         image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
         
         # show the prediction
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((28,28))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print("i think tha digit is : {}".format(prediction))
         #cv2.imshow("image", image)
         plt.show()
         cv2.waitKey(0)