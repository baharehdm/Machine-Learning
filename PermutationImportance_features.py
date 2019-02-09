#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import pandas as pd
import os
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display


# In[2]:


feature_df = pd.read_csv('P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\New feature dataset based on acceleration whole data\\new_feature_dataset_whole_original.csv')
IMU_class = pd.read_csv('P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\Class_dataset_whole.csv')

feature_df = feature_df.drop("Unnamed: 0", axis=1)
IMU_class = IMU_class.drop("Unnamed: 0", axis=1)


# In[3]:


x_train = feature_df.iloc[0:600349 , :]
x_test = feature_df.iloc[600349: , :]

y_train = IMU_class.iloc[0:600349 , :]
y_test = IMU_class.iloc[600349: , :]


# In[6]:


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


# In[7]:


#start = time.time()
#perm_rf = PermutationImportance(rf_model).fit(x_test, y_test)
#end = time.time()
#print(end - start)

with open('P:\\My Documents\\Thesis\\NEW Dataset classifications via cluster\\Feature Engineered\\Permutation\\RF_perm.pkl', 'rb') as f:
    perm_rf = pickle.load(f)


# In[10]:


eli5_perm_rf = eli5.show_weights(perm_rf, feature_names = x_test.columns.tolist())


# In[11]:


eli5.show_weights(perm_rf, feature_names = x_test.columns.tolist(), top = None )


# In[12]:


import eli5
from eli5.sklearn import PermutationImportance


# In[13]:


with open('P:\\My Documents\\Thesis\\NEW Dataset classifications via cluster\\Feature Engineered\\Permutation\\XGB_perm.pkl', 'rb') as f:
    perm_xgb = pickle.load(f)


# In[14]:


eli5.show_weights(perm_xgb, feature_names = x_test.columns.tolist(), top = None)


# In[15]:


with open('P:\\My Documents\\Thesis\\NEW Dataset classifications via cluster\\Feature Engineered\\Permutation\\log_perm.pkl', 'rb') as f:
    perm_log = pickle.load(f)


# In[16]:


eli5.show_weights(perm_log, feature_names = x_test.columns.tolist(), top= None)


# In[55]:


# knearest neighbour
from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")


# In[56]:


start = time.time()
k_model.fit(x_train, y_train)

end = time.time()
print(end - start)


# In[57]:


start = time.time()
perm_k = PermutationImportance(k_model).fit(x_test, y_test)

end = time.time()
print(end - start)


# In[58]:


eli5.show_weights(perm_k, feature_names = x_test.columns.tolist(), top = None)


# In[17]:


with open('P:\\My Documents\\Thesis\\NEW Dataset classifications via cluster\\Feature Engineered\\Permutation\\LDA_perm.pkl', 'rb') as f:
    perm_lda = pickle.load(f)


# In[18]:


eli5.show_weights(perm_lda, feature_names = x_test.columns.tolist(), top = None)


# In[19]:


with open('P:\\My Documents\\Thesis\\NEW Dataset classifications via cluster\\Feature Engineered\\Permutation\\extra_perm.pkl', 'rb') as f:
    perm_extra = pickle.load(f)


# In[20]:


eli5.show_weights(perm_extra, feature_names = x_test.columns.tolist(), top = None)


# In[ ]:




