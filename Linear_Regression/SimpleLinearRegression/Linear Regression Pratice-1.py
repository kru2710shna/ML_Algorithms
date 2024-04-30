#!/usr/bin/env python
# coding: utf-8

# In[20]:


#imports 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


# Datset
df=pd.read_csv("/Users/krushna/Downloads/height-weight.csv")
df


# In[22]:


# Scatter Plot to get relationship between Dependent and Independed Variable 
plt.scatter(df['Weight'] , df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("Weight VS Height Plot")


# In[23]:


# Correlation Matrix- To recognise relation +ive and -ive relationship 
df.corr()


# In[24]:


# Divide the data in dependent and Independent columns
df_features = df[['Weight']]
df_target = df['Height']


# In[25]:


# Train Test Split 
import sklearn 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.25, random_state=42)


# In[26]:


# Standardization
from sklearn.preprocessing import StandardScaler


# In[27]:


scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)


# In[28]:


# Apply Machine Learning Algo 
from sklearn.linear_model import LinearRegression
regression = LinearRegression(n_jobs =-1)


# In[29]:


regression.fit(df_features,df_target )


# In[30]:


# Predict
print("Coefficient or slope:",regression.coef_)
print("Intercept:",regression.intercept_)


# In[31]:


# Best Fit Line
plt.scatter(x_train,y_train)
plt.plot(x_train,regression.predict(x_train))


# ### Prediction of test data
# 

# In[34]:


# Predicted height output= intercept +coef_(Weights)
y_pred_test = 83.3 + 1.016*(x_test)
y_pred_test


# In[36]:


## Prediction for test data
y_pred=regression.predict(x_test)
y_pred


# In[37]:


## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[38]:


# Performace Metrices
#R^2

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)


# In[41]:


## OLS Linear Regression
import statsmodels.api as sm
model=sm.OLS(y_train,x_train).fit()
prediction=model.predict(x_test)
print(prediction)
print(model.summary())


# In[42]:


## Prediction For new data
regression.predict(scaler.transform([[72]]))


# In[ ]:




