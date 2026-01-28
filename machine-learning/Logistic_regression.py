import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


df=pd.read_csv(r"D:\AI_ML\Expert-Generative-AI-and-Agenetic-Ai-developer\machine-learning\logit classification.csv")


X=df.iloc[:, [2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, 
                                                  test_size=0.25, 
                                                  random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred) # Actual vs Predicted
print(cm) 

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)


























