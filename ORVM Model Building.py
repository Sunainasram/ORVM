#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


orvm_data=pd.read_csv("C:/Users/sunaina.s.ram/Downloads/ORVM Dataset.csv")


# In[3]:


orvm_data


# In[4]:


orvm_data.info()


# In[5]:


#data={'cust_rating':['Bad','Average','Good']}


# In[ ]:


#df_ordinal=pd.DataFrame(data)


# In[ ]:


#df_ordinal


# In[6]:


orvm_data


# In[7]:


orvm_data.head()


# In[8]:


orvm_data=orvm_data.dropna()
orvm_data


# In[9]:


# Import label encoder
from sklearn.preprocessing import LabelEncoder

# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

# Encode labels in column 'species'.
orvm_data['Vendor']= label_encoder.fit_transform(orvm_data['Vendor'])


# In[10]:


orvm_data['Vendor'].unique()


# In[11]:


orvm_data['Vehicle Type ']= label_encoder.fit_transform(orvm_data['Vehicle Type '])
orvm_data['Vehicle Type ']


# In[ ]:





# In[12]:


orvm_data['ORVM Type']=label_encoder.fit_transform(orvm_data['ORVM Type'])


# In[13]:


orvm_data['ORVM Type']


# In[14]:


orvm_data['Auto Dimming ']=label_encoder.fit_transform(orvm_data['Auto Dimming '])
orvm_data['Auto Dimming ']


# In[15]:


orvm_data.dtypes


# In[16]:


orvm_data['Vendor'].unique()


# In[17]:


orvm_data['Vendor'].value_counts()


# In[18]:


orvm_data['Vendor'].value_counts().sum()


# In[19]:


orvm_data=orvm_data.dropna()


# In[20]:


orvm_data


# In[94]:


x=orvm_data.iloc[:,4:7]


# In[95]:


x


# In[ ]:





# In[96]:


y=orvm_data.iloc[:,-1]


# In[97]:


y


# In[98]:


orvm_data


# In[99]:


orvm_data['Vendor'].unique()


# In[100]:


orvm_data.shape


# In[101]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import metrics


# In[102]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# model = svm.SVC() #select the algorithm
# model.fit(x_train,y_train) # we train the algorithm with the training data and the training output
# prediction=model.predict(x_test) #now we pass the testing data to the trained algorithm
# print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))#now we check the accuracy of the algorithm. 
# #we pass the predicted output by the model and the actual output

# model = LogisticRegression()
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test))

# In[103]:


modeld=DecisionTreeClassifier(criterion='gini', random_state=100,max_leaf_nodes=15)
modeld.fit(x_train,y_train)
prediction=modeld.predict(x_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test)*100)


# In[ ]:





# In[104]:


import matplotlib.pyplot as plt


# In[105]:


import graphviz 
import matplotlib.pyplot as plt


# In[106]:


tree.plot_tree(modeld)


# In[108]:


fn=['Price','Lead Time','Quality']
cn=['Forvia','Ficosa Internacional SA','Magna International Inc.','Mitsuba Corp','Murakami Corporation','Murakami Corporation','A1','ABC','XYZ','G Corporation','RK','Suppliers']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,30), dpi=200)
tree.plot_tree(modeld,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:





# In[109]:


def prediction(x_test, modeld):
  
    # Predicton on test with giniIndex
    prediction1=modeld.predict(x_test)
    print("Predicted values:")
    print(prediction1)
    return prediction1


# In[110]:


prediction(x_test, modeld)


# In[111]:


# Function to calculate accuracy
def cal_accuracy(y_test, prediction1):
    
    print("Report : ",
    classification_report(y_test, prediction1))


# In[112]:


cal_accuracy(y_test,prediction1)


# In[ ]:





# In[ ]:





# In[ ]:




