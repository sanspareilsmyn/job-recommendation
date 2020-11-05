# 201105_PCA
#!/usr/bin/env python
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


job_data = job_data.rename(columns = tag_dic)


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
df['job_companySize'].replace({"1-10": 1, "11-50": 2, "51-100": 3, "101-200": 4, "201-500": 5, "501-1000": 6, "1000 이상": 7}, inplace=True)
df['job_companySize'] = df['job_companySize'].fillna(2)
from numpy import int64
df['job_companySize'] = df['job_companySize'].astype(int64)


# In[18]:


df


# # 2. Feature Engineering

# In[44]:


X = df.drop(['applied', 'userID', 'jobID'], axis=1)
y = df['applied']


# In[45]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# ## a. Dimensionality Reduction

# PCA, LDA, t-SNE

# In[48]:


from sklearn.decomposition import PCA


# In[49]:


pca_train = PCA(n_components=50)
pca_train.fit(X_train)


# In[50]:


X_train = pca_train.transform(X_train)


# In[51]:


X_train.shape


# In[52]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)


# In[53]:


pca_valid = PCA(n_components=50)
pca_valid.fit(X_valid)


# In[55]:


X_valid = pca_valid.transform(X_valid)


# In[56]:


pred = clf.predict(X_valid)


# In[57]:


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(pred, y_valid)


# In[58]:


acc_score

