# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 18:14:33 2026

@author: Aman Deep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv(r"D:\AI_ML\Expert-Generative-AI-and-Agenetic-Ai-developer\machine-learning\Salary_Data.csv")

x=df.iloc[:, :-1]
y=df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

comparison= pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(comparison)


plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

print(regressor)

m_slope=regressor.coef_
print(m_slope)


c_intercept=regressor.intercept_
print(c_intercept)

emp_15=m_slope * 15 + c_intercept
print(emp_15)

emp_20=m_slope * 20 + c_intercept
print(emp_20)

#--------class of 21st jan 2025---------


bias=regressor.score(X_train, y_train)
print("Bias Score is ", bias)

variance= regressor.score(X_test, y_test)
print("Variance is ", variance)



df.mean()

df["Salary"].mean()


df.median()
df["Salary"].median()

df["Salary"].mode()


df.var()  # Sample

df.std()
df["Salary"].std()

#Coef of variance

from scipy.stats import variation
variation(df.values)


variation(df.Salary)


df.corr()

df["Salary"].corr(df["YearsExperience"])


df.skew()
df["Salary"].skew()


df.sem() #Standard Error

#from scipy.stats import stats

#df.apply(stats.zscore)



# Degree of Freedom 

y_mean=np.mean(y)
y_mean


ssr=np.sum((y_pred-y_mean)**2)
print("ssr is ", ssr)

y=y[0:6]
sse=np.sum((y-y_pred)**2)
print("sse is ", sse)

r_square=1-ssr/sse

print("r square value is", r_square)

import pickle 


filename='linear_regression_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as Linear_regression_model.pkl")


import os
print(os.getcwd)





    


