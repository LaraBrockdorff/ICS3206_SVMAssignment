print("Starting SVM Digit Recogntion for ICS 3206")
print("*importing libraries")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
print("*Loading data")
mnist = fetch_openml('mnist_784', version=1, cache=True)

X = pd.DataFrame(mnist.data)
Y= pd.Series(mnist.target).astype('int').astype('category')

cleanTestX = X.tail(2000)
cleanTestY = Y.tail(2000)

#Taking the first 100000 of the data set to make easier computation
X = X.head(10000)
Y = Y.head(10000)

#interpreting the data
print("X data")
print(X.shape)
# keep in mind 28  by 28 pixel values
print ("Y data")
print(Y.shape)
#y / output only needs one colum saying which class the image fell into

print("*splitting test and train")
#Splitting test and train
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.20)

print("** RUNNING SVM WITH DEFAULT LINEAR PARAMS")
##  Testing out linear kernal
svcLinear = SVC(kernel= 'linear')
svcLinear.fit(X_train,y_train)

predicted_Linear =svcLinear.predict(cleanTestX)
from sklearn.metrics import classification_report, accuracy_score

print("default linear results \n"+classification_report(cleanTestY,predicted_Linear))
accuracy_score(cleanTestY, predicted_Linear)



print("** RUNNING SVM WITH DEFAULT POLYNOMIAL PARAMS")
## Testing Polynomial Kernel
svcPoly = SVC(kernel= 'poly' )
svcPoly.fit(X_train,y_train)

predicted_Poly =svcPoly.predict(cleanTestX)

print("default poly results \n"+classification_report(cleanTestY,predicted_Poly))
accuracy_score(cleanTestY, predicted_Poly)



print("** RUNNING SVM WITH DEFAULT RBF PARAMS")
## Testing RBF kernel
svcRBF = SVC(kernel= 'rbf' )
svcRBF.fit(X_train,y_train)

predicted_RBF =svcRBF.predict(cleanTestX)

print("default rbf results \n"+classification_report(cleanTestY,predicted_RBF))
accuracy_score(cleanTestY, predicted_RBF)


print("** RUNNING SVM WITH DEFAULT SIGMOID PARAMS")
## Testing Sigmoid kernel
svcSigmoid = SVC(kernel= 'sigmoid' )
svcSigmoid.fit(X_train,y_train)

predicted_Sigmoid =svcSigmoid.predict(cleanTestX)

print("default sigmoid results \n"+classification_report(cleanTestY,predicted_Sigmoid))
accuracy_score(cleanTestY, predicted_Sigmoid)


### NOW TO FIND A MORE STRUCTURED APPROACH TO TUNING BY MAKING USE OF A GRID SEARCH
print("Making use of grid search to finder better hyper parameters")
print("*importing GridSearchCV")
from sklearn.model_selection import GridSearchCV

print("Starting tuning of RBF hyper paramters")
# defining parameter range
param_grid = {'C': [1, 2, 3],
              'gamma': [0.0000002, 0.0000003, 0.0000004, 0.00000044],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)

predicted_yNew =grid.predict(X_val)
print(accuracy_score(y_val, predicted_yNew))
print("****** ------- ********")
print(classification_report(y_val,predicted_yNew))

print(grid.best_estimator_)



print("Starting tuning of Polynomial degree hyper paramter")

# defining parameter range
param_gridPoly = {'degree': [2, 3, 4, 5],
                  'kernel': ['poly']}

gridP = GridSearchCV(SVC(), param_gridPoly, refit=True, verbose=3)

gridP.fit(X_train, y_train)

predicted_yP =gridP.predict(X_val)
print(accuracy_score(y_val, predicted_yP))
print("****** ------- ********")
print(classification_report(y_val,predicted_yP))

print(gridP.best_estimator_)


#On Test data not on validation
predicted_yPFinal =gridP.predict(cleanTestX)
print("Poly Trained results (results from test )\n "+classification_report(cleanTestY,predicted_yPFinal))

#On Test data not on validation
predicted_yRBGFinal =grid.predict(cleanTestX)
print("RBG Trained results (results from test ) \n "+classification_report(cleanTestY,predicted_yRBGFinal))
