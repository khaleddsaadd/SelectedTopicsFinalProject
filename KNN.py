import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("bank-full.csv")
#Converting classes into Numeric Values
df['housing'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)
df['loan'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)
df['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur"
,"student","blue-collar","self-employed","retired","technician","services"],
[1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)
df['marital'].replace(["married","divorced","single"],[1,2,3],inplace=True)
df['education'].replace(["unknown","secondary","primary","tertiary"],[0,1,2,3],inplace=True)
df['default'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)
df['contact'].replace(["unknown","telephone","cellular"],[0, 1,2], inplace=True)
df['poutcome'].replace(["unknown","other","failure","success"],[0,1,2,3],inplace=True)
df['y'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)

print(df.head)

# 0: age, 1: job, 3:education
X = df.iloc[:, [0, 1, 3]].values
y = df.iloc[:, 16].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#scaling data
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
print(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)


    #  ConfusionMatrix{ TruePositive     FalsePositive
    #                   FalseNegative    TrueNegative    }

Confusion_Matrix = confusion_matrix(y_test, y_pred)
TP = Confusion_Matrix[0][0]  #TruePositive
FP = Confusion_Matrix[0][1]  #FalsePositive
FN = Confusion_Matrix[1][0]  #FalseNegative
TN = Confusion_Matrix[1][1]  #TrueNegative


print(pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True),"\n")

print("ConfusionMatrix \n",Confusion_Matrix,"\n")


Accuracy = (TP + TN) / (TP + FP + TN +FN)
print(Accuracy,"\n")


Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 *( (Precision * Recall) /(Precision + Recall) )
print("Precision: \n", Precision)
print("Recall: \n", Recall)
print("F1: \n", F1_Score)