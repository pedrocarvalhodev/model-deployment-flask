
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestClassifier


# In[21]:


df = pd.read_csv("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-California-Housing-Prices/master/Data/housing.csv")
df = df.sample(frac=0.2)
df.shape


# In[22]:


df = df.dropna(axis=0)
df.drop(["longitude", "latitude"], axis = 1, inplace=True)


# In[23]:


df.groupby("ocean_proximity").median_house_value.count()


# In[24]:


print(df.shape)
df = df.merge(pd.get_dummies(df.ocean_proximity, drop_first=True, prefix="OCEAN_PROX"), 
              left_index=True, right_index=True, how="inner")
df.drop("ocean_proximity", axis = 1, inplace=True)
print(df.shape)


# In[25]:


df.head(2)


# In[26]:


#df.info()


# In[27]:


#df.columns


# In[28]:


y = "median_house_value"
X = [x for x in df.columns if x != y]

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[29]:


reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
y_pred = reg.predict(X_test)
yh  = [x for x in zip(y_test, y_pred)]
#print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE All: ", rootMeanSquaredError)


# ## Variable Importance

# In[30]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
df_X = df[X].copy()
df_X['randomVar'] = np.random.randint(1, 10, df_X.shape[0])
clf = clf.fit(df_X, df[y])
features = pd.DataFrame()
features['feature'] = df_X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features = features.sort_values(by="importance", ascending=False).reset_index(drop=False)
features


# In[31]:


randomVarIndex = features[features.feature=="randomVar"].index.values[0]


# In[32]:


feat_positive = list(features[features.index < randomVarIndex].feature.values)
feat_positive


# In[33]:


reg = LinearRegression(fit_intercept=True).fit(X_train[feat_positive], y_train)
y_pred = reg.predict(X_test[feat_positive])
yh  = [x for x in zip(y_test, map(int, y_pred))]
#print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE Better than random: ", rootMeanSquaredError)


# In[34]:


# Compare variable importance with predictive capacity of each var with intercept, Mean RMSE with train-test loop 


# ## Linear regression brute force eval

# In[35]:


y = "median_house_value"
X = [x for x in df.columns if x != y]


# In[36]:


X


# In[37]:


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
        for rs in range(10):
            X_train, X_test, y_train, y_test = train_test_split(df[x], df[y], test_size=0.20, random_state=rs)
            reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        res.append((x, np.mean(rmse)))
    res = pd.DataFrame(res, columns=["var", "rmse"])
    res = res.sort_values(by="rmse").reset_index(drop=True)
    res["nvars"] = res["var"].apply(lambda x : len(x))
    return res


# In[38]:


r = split_fit_eval(df=df, y="median_house_value")


# In[39]:


r.head(10)


# In[40]:


r["var"].loc[0]


# In[41]:


#r["var"].loc[0]


# In[42]:


features


# ## Var Importance - Correlations

# In[43]:


y = "median_house_value"
df_corr = df.corr()
df_corr = df_corr[y]
df_corr = df_corr.reset_index(drop=False)
df_corr[y] = df_corr[y].apply(lambda x : abs(x))
df_corr.sort_values(by=y, ascending=False).reset_index(drop=True)


# ## Remove explinatory vars with high correlations 

# In[44]:


df_corr = df.corr().reset_index(drop=False)
df_corr


# In[45]:


df_corr_melt = pd.melt(df_corr, id_vars=['index'], value_vars=[col for col in df_corr.columns if col != "index" ])
df_corr_melt["value"] = df_corr_melt["value"].apply(lambda x : abs(x))
df_corr_melt = df_corr_melt.sort_values(by="value", ascending=False)
df_corr_melt = df_corr_melt.loc[df_corr_melt["index"]!=df_corr_melt["variable"], :]
df_corr_melt = df_corr_melt[df_corr_melt["value"]>0.5]
df_corr_melt

