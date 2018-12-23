#!/usr/bin/env python
# coding: utf-8

# https://github.com/pratos/flask_api

# In[2]:


import pandas as pd
import sklearn.datasets as datasets


# In[3]:


#df = data.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False)
#print(df.DESCR)
#data = pd.DataFrame(df.data, columns=df.feature_names)
#data["medianValue"] = df.target


# In[4]:


df = datasets.load_boston()


# In[5]:


print(df.DESCR)


# In[6]:


data = pd.DataFrame(df.data, columns=df.feature_names)
data["medianValue"] = df.target


# In[7]:


data.head(2)


# https://github.com/pratos/flask_api/blob/master/notebooks/ML%2BModels%2Bas%2BAPIs%2Busing%2BFlask.md

# In[8]:


data.info()


# In[ ]:





# In[ ]:





# In[ ]:




