# 201105_baseline_finalcsv
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


# # 2. Baseline Model with Logistic Regression

# In[19]:


X = df.drop(['applied', 'userID', 'jobID'], axis=1)
y = df['applied']


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)


# In[22]:


pred = clf.predict(X_valid)


# In[23]:


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(pred, y_valid)


# In[24]:


acc_score


# # 3. Predicting Test Data

# In[25]:


test.head()


# In[26]:


# user_data와 job_data 로부터 새로 test_df 만들어야 됨


# In[27]:


test_df = test.merge(user_data, on='userID', how='inner')
test_df = test_df.merge(job_data, on='jobID', how='inner')


# In[29]:


test_df = test_df.drop(['job_companyID'], axis=1)
test_df['job_companySize'].replace({"1-10": 1, "11-50": 2, "51-100": 3, "101-200": 4, "201-500": 5, "501-1000": 6, "1000 이상": 7}, inplace=True)
test_df['job_companySize'] = test_df['job_companySize'].fillna(2)
from numpy import int64
test_df['job_companySize'] = test_df['job_companySize'].astype(int64)


# In[30]:


test_df


# In[33]:


X_test = test_df.drop(['userID', 'jobID'], axis=1)


# In[34]:


X_test.shape


# In[36]:


pred_test = clf.predict(X_test)


# In[38]:


pred_test


# In[39]:


sample_output_job.head()


# In[40]:


my_submission = pd.DataFrame({'applied': pred_test})


# In[41]:


my_submission.head()


# In[48]:


import datetime
time = datetime.datetime.now().strftime("%m/%d/%Y")
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




