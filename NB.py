import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools 
from itertools import product
 

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
X=df.iloc[:,[0,1,2,3,4,5,6,7,11]].values
y = df.iloc[:, 16].values

#Print Attributes and classes
print(X)
print(y)


#Splitting Dataset to Train and Test 
#Test size in 20% of whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#scaling data
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
print(X_test)


#Naive Bayes Classifier
NaiveBayes = GaussianNB()
#Train Model using X and Y (Train)
NaiveBayes.fit(X_train, y_train)

#Test Model using Testing DataSet (20%) 
y_pred = NaiveBayes.predict(X_test)

#Construct Confusion Matrix
Confusion_Matrix = confusion_matrix(y_test, y_pred)

TP = Confusion_Matrix[0][0]  #TruePositive
FP = Confusion_Matrix[0][1]  #FalsePositive
FN = Confusion_Matrix[1][0]  #FalseNegative
TN = Confusion_Matrix[1][1]  #TrueNegative

#Print Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True),"\n")
print("ConfusionMatrix \n",Confusion_Matrix,"\n")

#calculate accuracy (All predicted right / all number of predictions)
Accuracy = (TP + TN) / (TP + FP + TN +FN)
print(Accuracy,"\n")
#Calculate Percision (Predicted true right / number all predicted true)
Precision = TP / (TP + FP)
print("Precision: \n", Precision)
#Calculate Recall (Predicted true right / number of all supposed to be true)
Recall = TP / (TP + FN)
print("Recall: \n", Recall)
#Calculate FScore 
F1_Score = 2 *( (Precision * Recall) /(Precision + Recall) )
print("F1: \n", F1_Score)

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest')

for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white")

    
plt.ylabel('True label (Recall)')
plt.xlabel('Predicted label (Precision)')
plt.title('Logistic Regression with TFIDF | Confusion Matrix')
plt.colorbar();