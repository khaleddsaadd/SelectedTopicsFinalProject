# -*- coding: utf-8 -*-
"""FINAL-LogisticRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_4mywNYWMJXqERk-LOImvaiBS0j6viVY
"""

import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("bank-full.csv")
df['y'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)

print(df.head)

x=df.iloc[:,[0,5]].values
y=df.iloc[:,16].values
print(x)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain , ytest = train_test_split(x,y,test_size=0.5,random_state=0)

from sklearn.preprocessing import StandardScaler

df["age"] = pd.to_numeric(df["age"], downcast="float")
df["balance"] = pd.to_numeric(df["balance"], downcast="float")
df["job"] = pd.to_numeric(df["balance"], downcast="float")

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)
  
print (xtrain[0:, :])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
ConfusionMatrix = confusion_matrix(ytest, y_pred)
  
print ("Confusion Matrix : \n", ConfusionMatrix)
#True Pos     False Pos
#False Neg    True Neg

TP = ConfusionMatrix[0][0]
FP = ConfusionMatrix[0][1]
FN = ConfusionMatrix[1][0]
TN = ConfusionMatrix[1][1]
Precision = TP / (TP+FP)
print("Precision= ",Precision)

Recall = TP / (TP + FN)
print("Recall= ",Recall)

F1_Score = 2 * ((Precision*Recall) / (Precision+Recall))
print("F1-Score= ",F1_Score)

from sklearn.metrics import accuracy_score
print ("Ready-made Accuracy= ", accuracy_score(ytest, y_pred))

MyAccuracy = (TP+TN) / (TP+FP+TN+FN)
print("Calculated Accuracy= ", MyAccuracy)

from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest

from matplotlib.colors import ListedColormap

# %matplotlib inline
# pd.crosstab(df.job,df.y).plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')