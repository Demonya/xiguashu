
# coding: utf-8

# In[1]:

import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[2]:

data = load_boston()
x = data.data[:,5]
y = data.target
x.shape,y.shape


# In[3]:

plt.figure(figsize=(8,4))
plt.scatter(x,y)
plt.show()


# In[4]:

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
x_train.shape,y_train.shape


# In[5]:

model = LinearRegression().fit(x_train.reshape(-1,1),y_train)


# In[6]:

y_pre = model.predict(x_test.reshape(-1,1))


# In[7]:

mean_squared_error(y_test,y_pre)


# In[8]:

plt.figure(figsize=(8,4))
plt.scatter(x,y,color = 'green')
plt.plot(x_test,y_pre,color = 'b',linewidth= 2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()




