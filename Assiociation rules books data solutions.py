# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:33:31 2022

@author: mythili N
"""
# Load the Data
import pandas as pd 
df = pd.read_csv("D:\\data science\\Assignments\\Association Rules\\book.csv")
df.shape
list(df)
df.head()
type(df)

# Label Encode for categorical data
df_new=pd.get_dummies(df)
df_new.head()

#Apriori Algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_books = apriori(df_new, min_support=0.1, use_colnames=True)
frequent_books

rules = association_rules(frequent_books, metric="lift", min_threshold=0.7)
rules

rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

# Histogram Plot
rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

# scatter Plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()
