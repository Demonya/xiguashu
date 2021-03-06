#!/usr/bin/env python
# coding: utf-8

# # coding: utf-8
# 
# # ## 性能度量
# # ### 错误率：分类错误的样本数占总体样本数的比例
# # ### 精度：分类正确的样本数占总体样本的比例
# # ### 查准率：真正例占预测样本正例的比例   P = TP/TP+FP
# # ### 查全率：真正例占实际样本正例的比例   R = TP/TP+FN 
# # ### 查准率与查全率是一对矛盾，trade-off。
# # ### PR曲线：查准率为纵轴，查全率为横轴。
# # ### BEP：平衡点（查准率=查全率）
# # ### F1 = 2*P*R/(P+R) 
# # ### Fβ = (1+β^2)*P*R/(β^2*P)+R  考虑β趋于无穷时只有R  考虑β趋于0时只有P 
# # ### 真正例率 = 查全率 = TP/(TP+FN)
# # ### 假正例率 = FP/（FP+TN）
# # ### ROC曲线：真正例率为纵轴，假正例率为横轴。
# # ### AUC：ROC曲线下方面积
# # 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,plot_precision_recall_curve,average_precision_score


# In[2]:


data,label = load_iris().data[50:150],load_iris().target[50:150]
label


# In[3]:


x_train,x_test,y_train,y_test = train_test_split(data,label,test_size = 0.3,random_state = 66)


# In[4]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_hat = lr.predict(x_test)
diff = y_hat - y_test
acc_score = len(diff[diff==0])/len(diff)
#精度
print('计算的精度为%.5f'%acc_score)
print('sklearn的精度为%.5f'%accuracy_score(y_test,y_hat))


# In[5]:



#计算混淆矩阵
TP,FN,FP,TN = 0,0,0,0
# print(y_hat[0])
for i in range(len(y_test)):
    if y_test[i] == y_hat[i] and y_test[i] == 1:
        TP += 1
    elif y_test[i] == y_hat[i] and y_test[i] ==2:
        TN += 1
    elif y_test[i] != y_hat[i] and y_test[i] == 1:
        FN += 1
    elif y_test[i] != y_hat[i] and y_test[i] == 2:
        FP += 1

p = TP/(TP+FP)
r = TP/(TP+FN)
F1 = 2*p*r/(p+r)

#混淆矩阵
print('计算混淆矩阵:',[TP ,FN ,FP ,TN],'\n')
print('sklearn 混淆矩阵:' , confusion_matrix(y_test,y_hat).ravel() ,'\n') #sklearn

print('sklearn F1score = ',f1_score(y_test,y_hat,average=None))
print('查准率= %.5f,查全率= %.5f,F1score= %.5f'%(p,r,F1))


# In[6]:


y_scores = lr.predict_proba(x_test)[:,1]
y_test.dtype,y_scores.dtype
# y_test.reshape(-1,1)


# In[7]:


p,r,th = precision_recall_curve(y_test,y_scores,pos_label=1)

pr_curve = plot_precision_recall_curve(lr, x_test, y_test)
average_precision = average_precision_score(y_test, y_scores)
pr_curve.ax_.set_title('Precision-Recall curve: '
                   '平均精度={0:0.2f}'.format(average_precision))





