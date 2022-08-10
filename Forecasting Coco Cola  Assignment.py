# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 09:38:43 2022

@author: Mythili N 
"""
# Load the Data
import pandas as pd
import numpy as np 
import seaborn as sns
df = pd.read_csv("D:\\data science\\Assignments\\Forecasting\\CocaCola_Sales_Rawdata.csv")
df.shape
list(df)
df.head()
type(df)
df.info()

dates=pd.date_range(start='1986',periods=42,freq='Q')
dates
df1=pd.DataFrame(dates)
df1
df['dates']=dates
df

df.drop(['Quarter'],axis=1,inplace=True)
df.Sales.plot()
df.dates.plot()
 
t=[]
for i in range(1,43):
    t.append(i)
t
df['t'] = pd.DataFrame(t)    
df['log_sal'] = np.log(df['Sales'])
df['t_sq'] = df['t']*df['t']
df
df["Date"] = pd.to_datetime(df.dates,format="%b-%y")
df["month"] =df.Date.dt.strftime("%b") 
df["year"] =df.Date.dt.strftime("%Y") 
df2 = df.copy()
df2 = pd.get_dummies(df, columns = ['month'])
df2

import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
sns.boxplot(x="dates",y="Sales",data=df)
plt.figure(figsize=(10,8))
heatmap_y_month = pd.pivot_table(data=df,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") 
list(df2)
list(df)
df2.shape
Train = df2.head(32)
Test = df2.tail(10)
df2.shape

# fitting the model by using train ,test and t,t_sq,log_psg,dummies
import statsmodels.formula.api as smf 

#Linear Model
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

#Exponential
Exp = smf.ols('log_sal~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#Quadratic 
Quad = smf.ols('Sales~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
#pred_Quad = pd.Series(Exp.predict(pd.DataFrame(Test[["t","t_square"]))) # we hve to verify
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality 
add_sea = smf.ols('Sales~month_Mar+month_Jun+month_Sep+month_Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month_Mar','month_Jun','month_Sep','month_Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_sq+month_Mar+month_Jun+month_Sep+month_Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['month_Mar','month_Jun','month_Sep','month_Dec','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality
Mul_sea = smf.ols('log_sal~month_Mar+month_Jun+month_Sep+month_Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_sal~t+month_Mar+month_Jun+month_Sep+month_Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Compare the results 
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

''' inference: after fitting the model rmse_add_sea_quad and rmse_mul_add_sea is better
 and 4 dummies created by month
'''
