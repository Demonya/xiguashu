#!/usr/bin/env python
# coding: utf-8

# ## 基本概念：
# ##  评估方法：是对学习器的泛化误差进行评估的实验估计方法，解决的是在有限数据集的情况下怎么做的问题。通俗的说就是如何划分训练集与测试集。
# ##  性能度量：是对学习器的泛化能力评估标准，解决的是如何评价学习器的问题。
# ## 留出法：将数据集划分为互斥的两个集合，一个用作训练集一个用作测试集。随机采样与分层采样均可。需注意的是尽量保证划分的训练集与测试集分布一致。避免因训练集与测试集分布不一致而导致的偏差。单次使用留出法得到的估计不是十分可靠，一种可行的方法是多次随机采样或分层采样求取均值作为评估结果。训练集与测试集的样本比例控制在2/3或4/5.
# ##   交叉验证：将样本划分为K个大小相似的集合，尽量保证各个集合的分布一致。取K-1个集合作为训练集，留下的一个集合作为测试集。则会有K组训练集与测试集，最终通过K组的估计均值作为评估结果。评估结果的稳定性及保真性取决于K的取值。
# ##   自助法：自助法解决了留出法及交叉验证真实训练集被划分为训练集合测试集的问题。有放回抽样得到的样本作为训练集，原训练集中未被抽取的36.8%样本作为测试集。但需要注意bootstrapping改变了原训练数据集的分布，引入了估计偏差。

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# # 留出法、交叉验证

# In[2]:


X,y = load_iris().data,load_iris().target


# In[3]:


X_scale = StandardScaler(X)
s_precision = []
ns_precision = []
cross_precision = []
#留出法
def split_data(x,y,model):
    for i in range(10):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,stratify=y,random_state = i)#分层采样
        x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,test_size=0.2,shuffle=True,random_state = i)#随机采样
        m = model.fit(x_train,y_train)
        m1 = model.fit(x_train1,y_train1)
        m2 = cross_val_score(model,x_train,y_train,scoring = 'accuracy',cv = 10)
        cross_precision.append(np.mean(m2))
        y_hat = m.predict(x_test)
        y_hat1 = m1.predict(x_test1)
        diff = y_hat - y_test
        s_precision.append(len(diff[diff==0])/len(diff))
        diff1 = y_hat1 - y_test1
        ns_precision.append(len(diff1[diff1==0])/len(diff1))
    return s_precision,ns_precision,cross_precision
#s_percision:分层采样 ns_precision:随机采样 cross_precision:交叉验证


# In[4]:


s,ns,cr = split_data(X,y,LogisticRegression())


# In[5]:


def plot_accuracy(n,s,c):
    plt.figure(figsize=(20,5))
    plt.plot(range(1,11),n,label='分层抽样精度')
    plt.plot(range(1,11),s,label='随机抽样精度')
    plt.plot(range(1,11),c,label='交叉验证精度')
    plt.legend(loc='best',fontsize=12)
    plt.xlabel('X',fontsize=12)
    plt.ylabel('Y',fontsize=12)
    plt.grid()
    plt.title('不同类型划分精度')
    plt.show()


# In[6]:


plot_accuracy(s,ns,cr)


# # bootsttapping

# In[7]:


iris = pd.DataFrame(np.hstack((X,y[:,np.newaxis])))


# In[8]:


n = iris.shape[0]

# np.random.seed(2) #设置随机种子，复现随机结果
def generate_data(train_set):
    for i in range(n):
        train_set.append(int(np.random.random()*n))
    return train_set


# In[9]:


bootstrapping_precision = []
def bootstrapping(data):
    for i in range(10):
        train_set = []
        train_set = generate_data(train_set)
        train_data = data.iloc[train_set,:]
        test_data = data.iloc[np.setdiff1d(data.index,train_data.index)]
        print('未抽取的样本数{}'.format(test_data.shape[0]))
        train_data_input,train_label_input = train_data.iloc[:,:4],train_data.iloc[:,4]
        test_data_input,test_label_input = test_data.iloc[:,:4],test_data.iloc[:,4]
        lr = LogisticRegression().fit(train_data_input.values,train_label_input.values)
        y_hat2 = lr.predict(test_data_input.values)
        acc = accuracy_score(test_label_input.values,y_hat2)
        bootstrapping_precision.append(acc)
    return bootstrapping_precision


# In[10]:


acc = bootstrapping(iris)
acc


# In[17]:


def plot_bootstrappig(acc):
    plt.figure(figsize=(20,5))
    plt.plot(range(1,11),acc,label = '自主抽样')
    plt.xlabel('X',fontsize=12)
    plt.ylabel('Y',fontsize=12)
    plt.legend(loc='best')
    plt.show()

plot_bootstrappig(acc)




