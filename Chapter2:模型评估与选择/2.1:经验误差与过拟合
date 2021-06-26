#!/usr/bin/env python
# coding: utf-8

# ## 基本概念：（部分个人理解）
# ### 错误率 = 分类错误得样本数/样本总数量
# ### 精度 = 分类正确的样本数/样本总数量 = 1- 错误率
# ### 误差：样本label真实值与预测label之间的差异
# ### 经验误差（训练误差）：学习器在训练样本上的误差
# ### 泛华误差：学习器在测试集上的误差。训练模型的最终目的是能够得到很好的泛化能力。
# ### 过拟合与欠拟合：
# ### 过拟合是指学习器对于训练集的学习能力过强，对于训练集样本个性化或非一般的特性也作为评判标准，使得学习器在训练集上有很好的效果，但测试集上表现极差。
# ### 欠拟合：学习器的能力不够强，未学习到训练样本中的一般性特征。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# In[2]:


def generate_data():
    np.random.seed(0)
    n_sample = 30
    X = np.sort(np.random.rand(n_sample))
    y = np.exp(X)*np.cos(X) + np.random.rand(n_sample)*0.1
    return X,y


# In[3]:


def pic(X,y):
    plt.figure(figsize=(20,4))
    degrees = [1,4,8,16]
    for i in range(len(degrees)):
        plt.subplot(1, len(degrees) ,i + 1)

        poly = PolynomialFeatures(degrees[i],include_bias=False)
        lr = LinearRegression()
        pipe = Pipeline([('ploy',poly),('lr',lr)])
        pipe.fit(X[:,np.newaxis],y)
        
        #拟合曲线
        plt.plot(X,pipe.predict(X[:,np.newaxis]),label="model_predict")
        plt.plot(X,np.exp(X)*np.cos(X),label="real_func")
        plt.scatter(X,y,label="sample")
        plt.legend(loc = 'best',fontsize = 12)
        plt.xlabel('X',fontsize = 12)
        plt.ylabel('Y',fontsize = 12)
        plt.grid()
        plt.title('Degrees %d' %degrees[i])


# In[4]:


a,b = generate_data()


# In[5]:


pic(a,b)






