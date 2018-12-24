
# coding: utf-8

# In[62]:


import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv")


# In[3]:


df.head(2)


# In[4]:


df.drop("model", axis=1, inplace=True)


# In[5]:


#pd.get_dummies(df.mod
df.head(2)


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


print(df.nunique())
print(df.groupby("cyl").mpg.count())


# In[9]:


y = "hp"
X = [x for x in df.columns if x != y]

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.30, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[10]:


reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
y_pred = reg.predict(X_test)
yh  = [x for x in zip(y_test, y_pred)]
print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print(rootMeanSquaredError)


# ## Variable Importance

# In[11]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
df_X = df[X].copy()
df_X['randomVar'] = np.random.randint(1, 6, df_X.shape[0])
clf = clf.fit(df_X, df[y])
features = pd.DataFrame()
features['feature'] = df_X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features = features.sort_values(by="importance", ascending=False).reset_index(drop=False)
features


# In[12]:


randomVarIndex = features[features.feature=="randomVar"].index.values[0]


# In[13]:


feat_positive = list(features[features.index < randomVarIndex].feature.values)
feat_positive


# In[14]:


reg = LinearRegression(fit_intercept=True).fit(X_train[feat_positive], y_train)
y_pred = reg.predict(X_test[feat_positive])
yh  = [x for x in zip(y_test, map(int, y_pred))]
print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print(rootMeanSquaredError)


# In[15]:


# Compare variable importance with predictive capacity of each var with intercept, Mean RMSE with train-test loop 


# ## Linear regression brute force eval

# In[16]:


y = "hp"
X = [x for x in df.columns if x != y]


# In[17]:


X


# In[99]:


def split_fit_eval(df, y):
    X = [x for x in df.columns if x != y]
    res = []
    elements = np.arange(2,len(X)+1,1)
    ucombin=[]
    for e in elements:
        ucombin.append(list(itertools.combinations(X, e)))
    comb_flat_list = [list(item) for sublist in ucombin for item in sublist]
    for enum, x in enumerate(comb_flat_list):
        if enum % 100 == 0:
            print(enum)
        rmse = []
        df_X = df[x].copy()
        df_X["intercept"] = 1.0
        for rs in range(10):
            X_train, X_test, y_train, y_test = train_test_split(df_X, df[y], test_size=0.30, random_state=rs)
            reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        res.append((x, np.mean(rmse)))
    res = pd.DataFrame(res, columns=["var", "rmse"])
    res = res.sort_values(by="rmse").reset_index(drop=True)
    return res


# In[100]:


r = split_fit_eval(df=df, y="hp")


# In[101]:


r.head(10)


# In[59]:


features


# ## Var Importance - Correlations

# In[98]:


df_corr = df.corr()
df_corr = df_corr["hp"]
df_corr = df_corr.reset_index(drop=False)
df_corr["hp"] = df_corr["hp"].apply(lambda x : abs(x))
df_corr.sort_values(by="hp", ascending=False).reset_index(drop=True)

