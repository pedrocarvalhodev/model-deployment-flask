
# coding: utf-8

# Steps
# 
# 0. `Fields & Objetives Spreadsheet` : create spreadsheet detailing each field (example business meaning), problem and opoortunity, work steps 
# 0. `Fields relationship` : Document each field relation in comparison with target field (linear, corr, ..)
# 1. `Fillna` : Check isnull(), "None", NA, NaN, & groupby fillna(median), by main groups (sex, age bracket, city)
# 2. `Categories` : determine categorial variables and binning [50-100], get dummies or replaces
# 3. `Text Features` : get distinct caracteristics, or replace to aggregate similar texts
# 4. `Outliers` : Substitute outliers Pedrcentile > .99 with percentile 0.99
# 5. `Data types` : All numeric and as matrix
# 6. `Normalize`: transform 0-100 most vars, or scaling
# 7. `Var importance` : rank importance and drop correlated with low importance( below random ). Compare with OLS predict accuracy
# 7. `Feature Eng` : How many features can we create with above random baseline? Sum top 5 as target
# 8. `Data leakage` : Determin format to predict and evaluate models
# 9. `Pipelines` : create process for automation raw_data -> clean -> features -> predict -> evaluate
# 10. `Review Best practices` : first impressions matters, consistency, descripiton and communication, explanation

# In[1]:


import os 
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# In[2]:


path="/home/pedro/repos/ml_web_api/flask_api"


# In[3]:


data = pd.read_csv(path+'/data/training.csv')


# # 1. Diagnose data

# In[4]:


data.head(2)


# In[5]:


for _ in data.columns:
    print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))


# In[6]:


var_w_missing = [c for c in data.columns if data[c].isnull().sum() > 0]
var_w_missing


# In[7]:


for _ in var_w_missing:
    print("List of unique labels for {}:::{}".format(_, list(set(data[_].dropna()))[:10]))


# In[8]:


# Fields with no missing data
[c for c in data.columns if c not in var_w_missing]


# #### Assumptions for missing data
# 
# Could also fillna with most commun in cohort groupby of non missing data
# 
# For the rest of the missing values:
# 
# - `Dependents`: Assumption that there are no dependents
# - `Self_Employed`: Assumption that the applicant is not self-employed
# - `Loan_Amount_Term`: Assumption that the loan amount term is median value
# - `Credit_History`: Assumption that the person has a credit history
# - `Married`: If nothing specified, applicant is not married
# - `Gender`: Assuming the gender is Male for the missing values
# 
# Before that we'll divide the dataset in train and test

# # A. Clean Data

# # B. Feature eng & Var Importance w Correlation

# # 2. Split into test & train

# In[9]:


X_pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

y_pred_var = 'Loan_Status'

X_train, X_test, y_train, y_test = train_test_split(data[X_pred_var], data[y_pred_var],                                                     test_size=0.25, random_state=42)


# ## 2B Cleaning example

# In[10]:


X_train['Dependents'] = X_train['Dependents'].fillna('0')
X_train['Self_Employed'] = X_train['Self_Employed'].fillna('No')
X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean()) # median


X_train['Credit_History'] = X_train['Credit_History'].fillna(1)
X_train['Married'] = X_train['Married'].fillna('No')
X_train['Gender'] = X_train['Gender'].fillna('Male')

X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean()) # median


# # 2C Replace keys

# In[11]:


label_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']

for _ in label_columns:
    print("List of unique labels {}:{}".format(_, set(X_train[_])))


# In[12]:


gender_values = {'Female' : 0, 'Male' : 1} 
married_values = {'No' : 0, 'Yes' : 1}
education_values = {'Graduate' : 0, 'Not Graduate' : 1}
employed_values = {'No' : 0, 'Yes' : 1}
property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
X_train.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,                 'Self_Employed': employed_values, 'Property_Area': property_values, 'Dependents': dependent_values}                , inplace=True)


# In[13]:


X_train.head(2)


# In[14]:


# Check types of each fields
X_train.dtypes


# In[15]:


# Check if no nulls
for _ in X_train.columns:
    print("The number of null values in:{} == {}".format(_, X_train[_].isnull().sum()))


# In[16]:


X_train.head(2)


# In[17]:


X_train.shape


# # 2D test -> predict data format

# In[18]:


dftest = pd.read_csv(path+'/data/test.csv', encoding="utf-8-sig")
dftest.head(2)


# In[19]:


"""Converting Pandas Dataframe to json
"""
dftest.loc[0:1]#.to_json(orient='records')


# In[20]:


dftest.loc[0:1].to_json(orient='records')


# In[21]:


type(dftest.loc[0:1].to_json(orient='records'))


# # 3. Preprocessing Cass
# 
# Get all clean methods into single method

# In[22]:


from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
        
        df = df[pred_var]
        
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        
        gender_values = {'Female' : 0, 'Male' : 1} 
        married_values = {'No' : 0, 'Yes' : 1}
        education_values = {'Graduate' : 0, 'Not Graduate' : 1}
        employed_values = {'No' : 0, 'Yes' : 1}
        property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
        dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
        df.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,                     'Self_Employed': employed_values, 'Property_Area': property_values,                     'Dependents': dependent_values}, inplace=True)
        
        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
        """
        
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self


# In[23]:


# Make sure it works
X_pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

y_pred_var = 'Loan_Status'

X_train, X_test, y_train, y_test = train_test_split(data[X_pred_var], data[y_pred_var],                                                     test_size=0.25, random_state=42)


# In[24]:


X_train.head(2)


# In[25]:


for _ in X_train.columns:
    print("The number of null values in:{} == {}".format(_, X_train[_].isnull().sum()))


# In[26]:


# Declare class
preprocess = PreProcessing()


# In[27]:


preprocess


# In[28]:


preprocess.fit(X_train)


# In[29]:


X_train_transformed = preprocess.transform(X_train)


# In[30]:


X_train_transformed.shape


# In[31]:


X_train_transformed[:2,:6]


# In[32]:


# Do same for test dataset
X_test_transformed = preprocess.transform(X_test)
X_test_transformed.shape


# In[33]:


X_test_transformed[0]


# # 4. Format Y dataset

# In[34]:


## Clean y dataset
y_test.head(2)


# In[35]:


y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()
y_test[:4]


# In[36]:


y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()


# # 5. Set ML classes : pipe and grid

# In[37]:


param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
             "randomforestclassifier__max_depth" : [None, 6, 8, 10],
             "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
             "randomforestclassifier__min_impurity_decrease": [0.1, 0.2, 0.3]}

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(PreProcessing(),
                     RandomForestClassifier())

from sklearn.model_selection import train_test_split, GridSearchCV

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)


# In[38]:


pipe


# In[39]:


grid


# # 6 Deplot ML on data

# In[40]:


# Make sure it works
X_pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

y_pred_var = 'Loan_Status'

X_train, X_test, y_train, y_test = train_test_split(data[X_pred_var], data[y_pred_var],                                                     test_size=0.25, random_state=42)

grid.fit(X_train, y_train)


# In[41]:


print("Best parameters: {}".format(grid.best_params_))


# In[42]:


print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# # 7. Fit

# In[43]:


# X_test is in its original format
# but because we use pipe in the grid class, the preprocessing is run before the predict method
X_test[:1]


# In[44]:


y_hat_predict = grid.predict(X_test)
y_hat_predict[:5]


# # 8. Variable Importance

# In[45]:


y_train_b = y_train.replace({"Y": 1, "N":0})


# In[46]:


y_train_b[:5]


# In[47]:


## X clean
X_train['Dependents'] = X_train['Dependents'].fillna('0')
X_train['Self_Employed'] = X_train['Self_Employed'].fillna('No')
X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean()) # median


X_train['Credit_History'] = X_train['Credit_History'].fillna(1)
X_train['Married'] = X_train['Married'].fillna('No')
X_train['Gender'] = X_train['Gender'].fillna('Male')

X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean()) # median

gender_values = {'Female' : 0, 'Male' : 1} 
married_values = {'No' : 0, 'Yes' : 1}
education_values = {'Graduate' : 0, 'Not Graduate' : 1}
employed_values = {'No' : 0, 'Yes' : 1}
property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
X_train.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,                 'Self_Employed': employed_values, 'Property_Area': property_values, 'Dependents': dependent_values}                , inplace=True)


# In[69]:


X_train.head(2)


# In[70]:


X_train['randNumCol_1'] = np.random.randint(1, 6, X_train.shape[0])
X_train['randNumCol_2'] = np.random.randint(10, 16, X_train.shape[0])


# In[71]:


X_train.head(2)


# In[72]:


#import numpy as np
#import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X_train, y_train_b)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

#for f in range(X_train.shape[1]):
#    print("%d. feature %d - %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

for f, col in enumerate(X_train.columns):
    print("%d. feature (%s) - %d (%f)" % (f + 1, X_train.columns[indices][f], indices[f], importances[indices[f]]))
    
print(range(X_train.shape[1]), importances[indices])


# ## New Variable Importance

# In[73]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X_train, y_train_b)


# In[74]:


features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


# In[75]:


features.sort_values(by="importance", ascending=False)


# In[68]:


X_train.shape


# In[51]:


X_train.head(2)


# In[52]:


X_train.columns


# In[53]:


X_train.columns[indices][0]


# # 8B Correlation Matrix

# In[54]:


X_train.corr()[(X_train.corr()>0.4) | (X_train.corr() < -0.4)]


# # 9. Save Pickel Model

# In[55]:


import dill as pickle
#filename = 'model_v1.pk'
filename = 'model_v2.pk'


# In[56]:


# 1. Save model
with open(path+'/flask_api/models/'+filename, 'wb') as file:
    pickle.dump(grid, file)


# In[57]:


# 2. Read model
with open(path+'/flask_api/models/'+filename ,'rb') as f:
    loaded_model = pickle.load(f)


# In[58]:


# 3. Test model
loaded_model.predict(X_test)[:5]

