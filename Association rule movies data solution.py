# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:25:58 2022

@author: Mythili N
"""
# Load the Data
import pandas as pd
df = pd.read_csv("D:\\data science\\Assignments\\Association Rules\\my_movies.csv")
df.shape
list(df)

# Drop the variables
df.drop(['V1','V2','V3','V4','V5',],axis=1,inplace=True)
df.head()

df_new=pd.get_dummies(df)
df_new.head()
 
#Apriori Algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_movies = apriori(df_new, min_support=0.1, use_colnames=True)
frequent_movies

rules = association_rules(frequent_movies, metric="lift", min_threshold=0.7)
rules
rules.shape
rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

# Histogram
rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

# ScatterPlot
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()

