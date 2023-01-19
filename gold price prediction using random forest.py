#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all dependencies


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[3]:


#Reading data
data=pd.read_csv("C:\\Users\\psykid\\Desktop\\sreedevi mam\\DATASETS\\gld_price_data.csv")
data


# In[5]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.info


# In[10]:


data.isnull().sum()


# In[11]:


#correlation: 1.positive correlation 2.negative correlation


# In[12]:


correlation=data.corr()


# In[13]:


#Heatmap for correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')


# In[14]:


print(correlation['GLD'])


# In[15]:


#distribution of gold price
sns.distplot(data['GLD'],color='green')


# In[31]:


#splitting the data


# In[20]:


x=data.drop(['Date','GLD'],axis=1)
x


# In[21]:


y=data['GLD']
y


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=2)


# In[24]:


regressor=RandomForestRegressor(n_estimators=100)


# In[25]:


# training the model
regressor.fit(X_train,Y_train)


# In[26]:


#evaluating model
# prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[27]:


print(test_data_prediction)


# In[28]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# In[29]:


#compare actual value and predicted values
Y_test = list(Y_test)


# In[30]:


plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




