#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality reduction


# In[10]:


df = pd.read_csv("sales_data_sample.csv",encoding='unicode_escape')


# In[11]:


df.head()


# In[12]:


df.shape 


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[17]:


df.dtypes


# In[26]:


df['COUNTRY'].unique()


# In[27]:


df['PRODUCTLINE'].unique()


# In[28]:


df['DEALSIZE'].unique()


# In[29]:


productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical columns. 
Dealsize = pd.get_dummies(df['DEALSIZE'])


# In[30]:


df = pd.concat([df,productline,Dealsize], axis = 1)


# In[31]:


df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE'] #Dropping Country too as there are alot of countries. 
df = df.drop(df_drop, axis=1)


# In[32]:


df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the datatype.


# In[33]:


df.drop('ORDERDATE', axis=1, inplace=True) #Dropping the Orderdate as Month is already included.


# In[34]:


df.dtypes #All the datatypes are converted into numeric


# In[35]:


distortions = [] # Within Cluster Sum of Squares from the centroid
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)   #Appeding the intertia to the Distortions 


# In[36]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[37]:


X_train = df.values #Returns a numpy array.


# In[38]:


X_train.shape


# In[39]:


model = KMeans(n_clusters=3,random_state=2) #Number of cluster = 3
model = model.fit(X_train) #Fitting the values to create a model.
predictions = model.predict(X_train) #Predicting the cluster values (0,1,or 2)


# In[40]:


unique,counts = np.unique(predictions,return_counts=True)


# In[41]:


counts = counts.reshape(1,3)


# In[42]:


counts_df = pd.DataFrame(counts,columns=['Cluster1','Cluster2','Cluster3'])


# In[43]:


counts_df.head()


# In[ ]:




