# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:47:07 2022

@author: Myhtili N
"""
# Load the Data
import pandas as pd
df = pd.read_csv("D:\\data science\\Assignments\\Decision Trees\\Company_Data.csv")
df.shape
list(df)
type(df)
df.head()

# Converting x variable to number
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['ShelveLoc1'] = LE.fit_transform(df['ShelveLoc'])
df['ShelveLoc1']
df['Urban1'] = LE.fit_transform(df['Urban'])
df['Urban1']
df['US1'] =  LE.fit_transform(df['US'])
df['US1']
list(df)
df.shape

# Drop the variable
df_new=df.drop(['ShelveLoc','Urban','US','ShelveLoc1','Urban1','US1'],axis=1)
df_new

label = df.iloc[:,11:14]
label

# Standardization
from sklearn.preprocessing import StandardScaler,LabelEncoder
Scaler = StandardScaler()
x_scale=Scaler.fit_transform(df_new)
x_scale

x_new = pd.DataFrame(x_scale)
x_new

df_new_1 = pd.concat([x_new,label],axis=1)
df_new_1
list(df_new_1)

 # Taking x and y variable
x = df_new_1.iloc[:,1:11]
x.shape
list(x)
x.ndim

y = df.iloc[:,0]
y.shape
y.mean()

# Y variable covert into continuous to categorical 
y1 = []
for i in range(0,400,1):
    if y.iloc[i,]>=y.mean():
        print('High')
        y1.append('High')
    else:
        print('Low')
        y1.append('Low')
y_new=pd.DataFrame(y1)
y_new=LabelEncoder().fit_transform(y1)

# Splitting the train and test data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_new,test_size=0.25,stratify=y_new,random_state=42)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Selecting the model 
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier(criterion ='entropy') 
classifier.fit(x_train,y_train)
print(f'Decision tree has {classifier.tree_.node_count} nodes with maximum depth {classifier.tree_.max_depth}.')

# Prediction
y_pred = classifier.predict(x_test) 
y_pred.shape

# Confusion matrix for accuracy score
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
acc=metrics.accuracy_score(y_test,y_pred).round(2)
print(acc)

# Visualization  Tree.Plot
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
Classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
model = Classifier.fit(x,y_new)

%matplotlib qt
fig = plt.figure(figsize=(16,9))
Class= tree.plot_tree(Classifier, feature_names=df_new_1, filled=True,fontsize=6)

'''
Inference:
          In DecisionTree models(Gini,Entropy) ,Entopy gives the best accuracy Score:71%
          Entropy is better then gini 
'''



