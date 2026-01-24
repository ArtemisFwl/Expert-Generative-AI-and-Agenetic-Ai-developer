# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:55:27 2026

@author: Aman Deep
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\AI_ML\Expert-Generative-AI-and-Agenetic-Ai-developer\machine-learning\Data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer()

imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,0]=le.fit_transform(x[:,0])

le_y=LabelEncoder()
y=le_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)