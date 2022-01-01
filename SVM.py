import pandas as pn
import numpy as ny
import matplotlib.pyplot as plot 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


#Importing our dataset (bank-full dataset)
bank_dataset = pn.read_csv("bank-full.csv")

#To display how many rows and the columns of our dataset (bank-full dataset)
# print(bank_dataset.shape)

#To display our dataset as (pandas.core.frame.DataFrame)
# print(bank_dataset.head())

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



#divide our bank dataset into training set and testing set
cls_train, cls_test, lbl_train, lbl_test = train_test_split(cls, lbl, test_size = 0.20)


#Train dataset
sclassifier = SVC(kernel='linear') #linear is used on simple SVM and it can only classify linearly separable data
sclassifier.fit(cls_train, lbl_train) #We use fit method of SVC to train the algorithm on our dataset

#Prediction
lbl_pred = sclassifier.predict(cls_test) #predict is a method of the SVC class that make prediction



#Display the confusion matrix
print(confusion_matrix(lbl_test,lbl_pred))
print(classification_report(lbl_test,lbl_pred))




