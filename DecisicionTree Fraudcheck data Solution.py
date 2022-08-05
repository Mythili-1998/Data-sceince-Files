# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:54:16 2022

@author: Mythili N
"""
# Load the Data
import pandas as pd
import seaborn as sns
data=pd.read_csv("D:\\data science\\Assignments\\Decision Trees\\Fraud_check.csv")
data.head()
list(data)
data.shape
data.dtypes
X1=data['Taxable.Income']
X1.shape
data.drop(['Taxable.Income'],axis=1,inplace=True)
data

X2=[]
for i in range(0,600,1):
    if X1.iloc[i,]<=30000:
        print('Risky')
        X2.append('Risky')
    else:
        print('Good')
        X2.append('Good')
X2
X2_new=pd.DataFrame(X2)
X2_new.set_axis(['Category'],axis='columns',inplace=True)
data_new=pd.concat([data,X2_new],axis=1)
data_new
list(data_new)
data_new.shape

sns.distplot(data_new['City.Population'])
sns.distplot(data_new['Work.Experience'])

sns.countplot(data_new['Undergrad'])
sns.countplot(data_new['Marital.Status'])
sns.countplot(data_new['Urban'])
sns.countplot(data_new['Category'])

#Splitting data into X and Y
X=data_new.iloc[:,0:5]
X.head()
X.shape
X.dtypes
Y=data_new['Category']
Y.head()

#Preprocessing the data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
X['Undergrad']=LE.fit_transform(X['Undergrad'])
X['Marital.Status']=LE.fit_transform(X['Marital.Status'])
X['Urban']=LE.fit_transform(X['Urban'])
Y=LE.fit_transform(Y)
print(X)
print(Y)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=99)
X_train.shape

#Decision tree Classifier (As Y have 2 outputs we choose Classifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
DT=DecisionTreeClassifier(criterion='entropy',max_depth=3).fit(X_train,Y_train)
Y_pred=DT.predict(X_test)
acc=accuracy_score(Y_test,Y_pred)
print((acc*100).round(3))

#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth
'''
Inference: For random state 99, by using Entropy criterion and for the max depth of 3 in decision tree we are achieving 
           the 80% accuracy for the given dataset.
'''




