import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"D:\AI_ML\Expert-Generative-AI-and-Agenetic-Ai-developer\machine-learning\Investment_data.csv")

df.head()

X=df.iloc[:, :-1]
y=df.iloc[:,-1]

X=pd.get_dummies(X, dtype=int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

print(y_pred)

m=regressor.coef_
print(m)

c=regressor.intercept_
print(c)

X=np.append(arr=np.full((50,1),42467).astype(int), values=X, axis=1)
# y=mx+c me c ke liye kiya hai ye
# Ihave appended the constant

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]

#Ordinary Lest Squares 

regressor_OLS =sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,5]]

#Ordinary Lest Squares 

regressor_OLS =sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
X_opt=X[:,[0,1,2,3]]

#Ordinary Lest Squares 

regressor_OLS =sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt=X[:,[0,1,2]]

#Ordinary Lest Squares 

regressor_OLS =sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()




bias=regressor.score(X_train, y_train)
print(bias)

variance=regressor.score(X_test, y_test)
print(variance)






    





















