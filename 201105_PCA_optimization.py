# 201105_PCA_optimization

# !/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


train = pd.read_csv('train_job/train.csv')
test = pd.read_csv('test_job.csv')

job_tags = pd.read_csv('train_job/job_tags.csv')
user_tags = pd.read_csv('train_job/user_tags.csv')
tags = pd.read_csv('train_job/tags.csv')
job_companies = pd.read_csv('train_job/job_companies.csv')

sample_output_job = pd.read_csv('sample_output_job.csv')

# # 1. Preprocessing DataFrame

# ## a. User Data

# In[3]:


user_data = user_tags.set_index(user_tags.userID)['tagID'].str.get_dummies().max(level=0)

# In[4]:


tag_dic = dict(tags.values)

# In[5]:


user_data = user_data.rename(columns=tag_dic)

# In[6]:


user_data = user_data.reset_index().rename(columns={"index": "userID"})

# In[7]:


user_data = user_data.add_prefix('user_')
user_data = user_data.rename(columns={'user_userID': 'userID'})

# In[8]:


user_data

# ## b. Job Data

# In[9]:


job_data = job_tags.set_index(job_tags.jobID)['tagID'].str.get_dummies().max(level=0)

# In[10]:


job_data = job_data.rename(columns=tag_dic)

# In[11]:


company_data = job_companies.set_index('jobID')

# In[12]:


job_data = pd.merge(job_data, company_data,
                    left_index=True, right_index=True, how='inner')

# In[13]:


job_data = job_data.reset_index().rename(columns={"index": "jobID"})

# In[14]:


job_data = job_data.add_prefix('job_')
job_data = job_data.rename(columns={'job_jobID': 'jobID'})

# In[15]:


job_data

# ## c. Merging into one integrated DataFrame

# In[16]:


df = train.merge(user_data, on='userID', how='inner')
df = df.merge(job_data, on='jobID', how='inner')

# In[17]:


df = df.drop(['job_companyID'], axis=1)
df['job_companySize'].replace(
    {"1-10": 1, "11-50": 2, "51-100": 3, "101-200": 4, "201-500": 5, "501-1000": 6, "1000 이상": 7}, inplace=True)
df['job_companySize'] = df['job_companySize'].fillna(2)
from numpy import int64

df['job_companySize'] = df['job_companySize'].astype(int64)

# In[18]:


df

# # 2. Feature Engineering

# In[19]:


X = df.drop(['applied', 'userID', 'jobID'], axis=1)
y = df['applied']

# In[20]:


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ## a. PCA Optimization

# In[23]:


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[26]:


def pca_optimization(X_train_, X_valid_, y_train_, y_valid_):
    acc_scores = {}
    n_candidates = [10, 20, 50, 100, 200, 300, 400, 500]
    for n_candidate in n_candidates:
        X_train, X_valid, y_train, y_valid = X_train_, X_valid_, y_train_, y_valid_
        pca_train = PCA(n_components=n_candidate)
        pca_train.fit(X_train)
        X_train = pca_train.transform(X_train)
        print(X_train.shape)

        clf = LogisticRegression(random_state=0).fit(X_train, y_train)

        pca_valid = PCA(n_components=n_candidate)
        pca_valid.fit(X_valid)
        X_valid = pca_valid.transform(X_valid)

        pred = clf.predict(X_valid)
        acc_score = accuracy_score(pred, y_valid)
        acc_scores[n_candidate] = acc_score

    return acc_scores


# In[28]:


acc_scores = pca_optimization(X_train, X_valid, y_train, y_valid)

# In[38]:


print(acc_scores)
print(acc_scores.keys())
print(acc_scores.values())

import seaborn as sns

g = sns.barplot(x=list(acc_scores.keys()), y=list(acc_scores.values()))

# In[ ]:




