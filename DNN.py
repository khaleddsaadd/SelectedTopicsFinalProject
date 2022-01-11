import pandas as pd
import seaborn as sns
import matplotlib as plt
import tf_slim as slim
import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv("bank-full.csv")
df['y'].replace(['no', 'yes' , 'unkown'],[0, 1,-1], inplace=True)
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
df['month'].replace(["unknown","other","failure","success"],[0,1,2,3],inplace=True)

df["age"] = pd.to_numeric(df["age"], downcast="float")
df["balance"] = pd.to_numeric(df["balance"], downcast="float")

df["job"] = pd.to_numeric(df["job"], downcast="float")
df["marital"] = pd.to_numeric(df["marital"], downcast="float")
df["education"] = pd.to_numeric(df["education"], downcast="float")
df["default"] = pd.to_numeric(df["default"], downcast="float")
df["housing"] = pd.to_numeric(df["housing"], downcast="float")

df["loan"] = pd.to_numeric(df["loan"], downcast="float")
df["contact"] = pd.to_numeric(df["contact"], downcast="float")
df["duration"] = pd.to_numeric(df["duration"], downcast="float")

df = df.drop('month', 1)

df.head(10)
print(df.head)

# %matplotlib inline
sns.countplot(data=df,x="y")
sns.pairplot(data=df,hue="y")

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop("y",axis=1))
scaled_fea = scaler.fit_transform(df.drop("y",axis=1))
df_1=pd.DataFrame(scaled_fea,columns=df.columns[:-1])
df_1.head()
X=df_1
y=df["y"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=2)
classifier.fit(X_train, y_train, steps=200, batch_size=20)
note_predictions = list(classifier.predict(X_test))



type(y_test)
Confusion_Matrix = confusion_matrix(y_test, note_predictions)
print(Confusion_Matrix)
TP = Confusion_Matrix[0][0]  #TruePositive
FP = Confusion_Matrix[0][1]  #FalsePositive
FN = Confusion_Matrix[1][0]  #FalseNegative
TN = Confusion_Matrix[1][1] 
Accuracy = (TP + TN) / (TP + FP + TN +FN)
print("Accuracy: ",Accuracy,"\n")

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 *( (Precision * Recall) /(Precision + Recall) )
print("Precision: \n", Precision)
print("Recall: \n", Recall)
print("F1: \n", F1_Score)