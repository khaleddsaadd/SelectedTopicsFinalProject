import pandas as pd
import seaborn as sns
import matplotlib as plt

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