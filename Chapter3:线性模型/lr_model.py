
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt 


# In[2]:

col = ['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)']
data = load_iris()
target_names = np.array(['setosa', 'versicolor', 'virginica'])
x,y = pd.DataFrame(data.data,columns=col),data.target


# In[3]:

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle = True,random_state = 1,stratify=y)
x_train.describe().T,x_test.describe().T


# In[4]:

model = LogisticRegression().fit(x_train,y_train)


# In[5]:

y_hat = model.predict(x_test)
confusion_matrix(y_test,y_hat)


# In[6]:

print(classification_report(y_test,y_hat,target_names=target_names))








