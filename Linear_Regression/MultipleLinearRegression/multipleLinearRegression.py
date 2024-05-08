#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[40]:


df= pd.read_csv("/Users/krushna/Downloads/economic_index.csv")


# In[41]:


df.head()


# In[42]:


updated_df = df.drop(columns=['Unnamed: 0', 'year','month'])


# In[43]:


# check null values 
updated_df.isnull().sum()


# In[44]:


# Visualize the data 
sns.pairplot(updated_df)


# In[45]:


# Corraltion amogst each other features variables
updated_df.corr()


# In[46]:


# -ve values means -ve slope/trend
# +ve values means +ve slope/trend
## Visualiza the datapoints more closely
plt.scatter(updated_df['interest_rate'],updated_df['unemployment_rate'],color='r')
plt.xlabel("Interest rate")
plt.ylabel("unemployment rate")


# In[47]:


# Independent and Dependent Variable 
X=updated_df.iloc[:,:-1]
y=updated_df.iloc[:,-1]


# In[49]:


# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[51]:


sns.regplot(x='interest_rate', y='unemployment_rate', data=updated_df)


# In[53]:


sns.regplot(x='interest_rate', y='index_price', data=updated_df)


# In[54]:


sns.regplot(x='interest_rate', y='index_price', data=updated_df)


# In[56]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[58]:


X_train


# In[59]:


X_test


# In[60]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)
## prediction
y_pred=regression.predict(X_test)


# In[61]:


## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[62]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
#display adjusted R-squared
print(1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))


# In[63]:


## OLS Linear Regression
import statsmodels.api as sm
model=sm.OLS(y_train,X_train).fit()


# In[64]:


model.summary()


# In[ ]:




