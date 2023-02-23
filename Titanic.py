#!/usr/bin/env python
# coding: utf-8

# In[111]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[112]:


data= pd.read_csv("Titanic.csv")
print(data)


# In[113]:


data.shape


# In[114]:


data.describe()


# In[115]:


data.info()


# In[116]:


data.head(10)


# In[117]:


print("number of passenger in data: "+str(len(data)))


# # Analyzing data

# In[118]:


sns.countplot(x="Survived",data=data)


# In[119]:


sns.countplot(x="Survived", hue="Sex",data=data)


# In[120]:


sns.countplot(x="Survived",hue="Pclass",data=data)


# In[121]:


data["Age"].plot.hist()


# In[122]:


data["Fare"].plot.hist()


# In[123]:


sns.countplot(x="SibSp",data=data)


# In[124]:


sns.countplot(x="Parch",data=data)


# In[125]:


sns.countplot(x="Embarked",data=data)


# # Data wranling

# In[126]:


data.isnull()


# In[127]:


data.isnull().sum()


# In[128]:


sns.heatmap(data.isnull())


# In[129]:


sns.boxplot(x="Pclass",y="Age",data=data)


# In[130]:


data.head(10)


# In[131]:


# removing NAN values from data
data.drop("Cabin",axis=1,inplace=True)


# In[132]:


data.dropna(inplace=True)


# In[133]:


sns.heatmap(data.isnull())


# In[134]:


data.isnull().sum()


# In[135]:


data.head(2)


# In[136]:


sex=pd.get_dummies(data["Sex"],drop_first=True)
sex.head(5)


# In[137]:


embark=pd.get_dummies(data["Embarked"],drop_first=True)
embark.head(5)


# In[138]:


pcl=pd.get_dummies(data["Pclass"],drop_first=True)
pcl.head(5)


# In[139]:


data=pd.concat([data,sex,embark,pcl],axis=1)


# In[140]:


data.drop(["Sex","PassengerId","Name","Ticket","Pclass","Embarked"],axis=1,inplace=True)


# In[141]:


data.head(5)


# # train data

# In[142]:


X=data.drop("Survived",axis=1)
y=data["Survived"]


# In[143]:


from sklearn.model_selection import train_test_split


# In[144]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=1)


# In[145]:


from sklearn.linear_model import LogisticRegression


# In[146]:


logmodel=LogisticRegression()


# In[147]:


logmodel.fit(X_train, y_train)


# In[148]:


predictions= logmodel.predict(X_test)


# In[149]:


from sklearn.metrics import classification_report


# In[150]:


classification_report(y_test,predictions)


# In[152]:


from sklearn.metrics import confusion_matrix


# In[153]:


confusion_matrix(y_test,predictions)


# In[154]:


from sklearn.metrics import accuracy_score


# In[155]:


accuracy_score(y_test,predictions)


# In[156]:


predictions


# In[ ]:




