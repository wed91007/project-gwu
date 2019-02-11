import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
# load pima dataset
# http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
# http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names
start = time.clock()
dataset = pd.read_csv('F:/Python/pima/diabetes.csv')

# Replace zeroes
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

# split dataset
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)
# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("training data points: {}".format(len(y_train)))
print("validation data points: {}".format(len(y_val)))
print("testing data points: {}".format(len(y_test)))

kVals = range(1, 30, 2)
accuracies = []
# Define the model: Init K-NN
for k in xrange(1, 30, 2):
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
#
## Fit Model
    classifier.fit(X_train, y_train)
    score = classifier.score(X_val, y_val)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))


model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(X_train, y_train)
classifier = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
classifier.fit(X_train, y_train)
# Predict the test set results
y_pred = classifier.predict(X_test)
 # save confusion matrix and slice into four pieces
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]   
print('TP=',TP,'TN=',TN)
print('FP=',FP,'FN=',FN)
# Evaluate Model
print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, y_pred))
print("Confusion matrix")
print (confusion)
elapsed = (time.clock()-start)
print("Time used:",elapsed)


print(f1_score(y_test, y_pred))
# STEP 5: Evaluate Model
#########################
# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))