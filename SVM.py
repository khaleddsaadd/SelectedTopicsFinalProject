from enum import auto
import pandas as pn
import numpy as ny
import matplotlib.pyplot as plot 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#Importing our dataset (bank-full dataset)
bank_dataset = pn.read_csv("bank-full.csv")

#To display how many rows and the columns of our dataset (bank-full dataset)
# print(bank_dataset.shape)

#To display our dataset as (pandas.core.frame.DataFrame)
print(bank_dataset.head())

labEn = LabelEncoder()
bank_dataset['job'] = labEn.fit_transform(bank_dataset['job'])
bank_dataset['marital']= labEn.fit_transform(bank_dataset['marital'])
bank_dataset['education']= labEn.fit_transform(bank_dataset['education'])
bank_dataset['default']= labEn.fit_transform(bank_dataset['default'])
bank_dataset['housing']= labEn.fit_transform(bank_dataset['housing'])
bank_dataset['loan']= labEn.fit_transform(bank_dataset['loan'])
bank_dataset['contact']= labEn.fit_transform(bank_dataset['contact'])
bank_dataset['month']= labEn.fit_transform(bank_dataset['month'])
bank_dataset['y']= labEn.fit_transform(bank_dataset['y'])
bank_dataset['poutcome']= labEn.fit_transform(bank_dataset['poutcome'])

#These 2 lines are to divide dataset into classes and labels
cls = bank_dataset.drop('y', axis=1) #drop the label and save classes in cls
lbl = bank_dataset['y'] #put labels only in y

# DataFrames = ny.arange(0, len(cls))

# plty=ny.array(cls)
# pltx=ny.array(lbl)

# print(len(pltx))
# print(len(plty))

# plot.title('Dataset')
# plot.plot(DataFrames, plty)
# plot.xlabel('DataFrames')
# plot.ylabel('Classes')
# plot.show()

#divide our bank dataset into training set and testing set
cls_train, cls_test, lbl_train, lbl_test = train_test_split(cls, lbl, test_size = 0.30)

sc = StandardScaler()
cls_train = sc.fit_transform(cls_train)
cls_test = sc.transform(cls_test)

#Train dataset
sclassifier = svm.SVC(kernel='rbf') #Gaussian kernel
sclassifier.fit(cls_train, lbl_train) #We use fit method of SVC to train the algorithm on our dataset

#Prediction
lbl_pred = sclassifier.predict(cls_test) #predict is a method of the SVC class that make prediction



plty=ny.array(cls_test)
pltx=ny.array(lbl_test)

print(len(pltx))
print(len(plty))

plot.title('Dataset')
plot.plot(pltx, plty , 'b.',label="Test Data")
plot.plot(lbl_pred, plty,'ro', label="Pred Data")
plot.xlabel('Labels')
plot.ylabel('Data Frames')
plot.show()

#Display the confusion matrix
print(confusion_matrix(lbl_test,lbl_pred))
print(classification_report(lbl_test,lbl_pred))





