#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

data_train = pd.read_csv("D:\\自我學習\\AI\\Kaggle\\Case 1_Titanic\\DataSet\\titanic\\train.csv")
data_train


# In[3]:


data_train.info() #檢視有哪些欄位空值的比例


# In[4]:


#6.1 乘客各屬性分佈
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2) # 設定圖表顏色alpha參數

plt.subplot2grid((2,3),(0,0)) # 在一張大圖里分列幾個小圖
data_train.Survived.value_counts().plot(kind='bar')# 柱狀圖
plt.title(u"獲救情況 (1為獲救)") # 標題
plt.ylabel(u"人數")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人數")
plt.title(u"乘客等級分佈")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年齡") # 設定縱坐標名稱
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年齡看獲救分佈 (1為獲救)")
plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年齡")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等級的乘客年齡分佈")
plt.legend((u'頭等艙', u'2等艙',u'3等艙'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人數")
plt.ylabel(u"人數")
plt.show()


# In[5]:


#6.2 屬性與獲救結果的關聯統計

#看看各乘客等級的獲救情況
fig = plt.figure()
fig.set(alpha=0.2) # 設定圖表顏色alpha參數

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'獲救':Survived_1, u'未獲救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等級的獲救情況")
plt.xlabel(u"乘客等級")
plt.ylabel(u"人數")
plt.show()


# In[3]:


data_test = pd.read_csv("D:\\自我學習\\AI\\Kaggle\\Case 1_Titanic\\DataSet\\titanic\\test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接著我們對test_data做和train_data中一致的特徵變換
# 首先用同樣的RandomForestRegressor模型填上丟失的年齡
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根據特徵屬性X預測年齡並補上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test


# In[ ]:




