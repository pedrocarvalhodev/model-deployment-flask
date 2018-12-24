#!/usr/bin/env python
# coding: utf-8

# https://github.com/pratos/flask_api

# In[19]:


import numpy as np
import pandas as pd
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


# In[3]:


#df = data.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False)
#print(df.DESCR)
#data = pd.DataFrame(df.data, columns=df.feature_names)
#data["medianValue"] = df.target


# In[4]:


df = datasets.load_boston()


# In[29]:


#print(df.DESCR)


# In[33]:


data = pd.DataFrame(df.data, columns=df.feature_names)
data["medianValue"] = df.target


# In[34]:


#data.head(2)


# https://github.com/pratos/flask_api/blob/master/notebooks/ML%2BModels%2Bas%2BAPIs%2Busing%2BFlask.md

# In[35]:


y = "medianValue"
X = [x for x in data.columns if x != y]


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[37]:


X_train.columns


# In[32]:


#X_train.describe()


# In[18]:


#down_quantiles = df.quantile(0.05)
#outliers_low = (df < down_quantiles)
#df.mask(outliers_low, down_quantiles, axis=1)


# In[27]:


X_train_norm = (X_train - X_train.mean()) / (X_train.max() - X_train.min())
X_train_norm = X_train_norm.apply(lambda x : np.around(x,1))
X_train_norm.columns = [x+"_norm" for x in X_train.columns]
X_train = X_train.merge(X_train_norm, how='inner', left_index=True, right_index=True)


# In[28]:


X_train.head(2)


# In[ ]:





# In[ ]:





# # test model

# In[41]:


import pandas as pd
from sklearn.externals import joblib
model = joblib.load('/home/pedro/repos/ml_web_api/model-deployment-flask/models/pipe_model_boston.pkl')
res = model.predict("/home/pedro/repos/ml_web_api/model-deployment-flask/data/boston_housing/X_test.csv")
print(res)


# In[42]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




