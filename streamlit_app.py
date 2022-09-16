#!/usr/bin/env python
# coding: utf-8
import pandas as pd
orvm_data=pd.read_csv("C:/Users/sunaina.s.ram/Downloads/ORVM Dataset.csv")




orvm_data



orvm_data.info()



orvm_data





orvm_data.head()

orvm_data=orvm_data.dropna()
orvm_data

# Import label encoder
from sklearn.preprocessing import LabelEncoder

# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

# Encode labels in column 'species'.
orvm_data['Vendor']= label_encoder.fit_transform(orvm_data['Vendor'])

orvm_data['Vendor'].unique()
orvm_data['Vehicle Type ']= label_encoder.fit_transform(orvm_data['Vehicle Type '])
orvm_data['Vehicle Type ']


orvm_data['ORVM Type']=label_encoder.fit_transform(orvm_data['ORVM Type'])
orvm_data['ORVM Type']

orvm_data['Auto Dimming ']=label_encoder.fit_transform(orvm_data['Auto Dimming '])
orvm_data['Auto Dimming ']
orvm_data.dtypes
orvm_data['Vendor'].unique()
orvm_data['Vendor'].value_counts()
orvm_data['Vendor'].value_counts().sum()


# In[ ]:


orvm_data=orvm_data.dropna()


# In[ ]:


orvm_data


# In[ ]:


x=orvm_data.iloc[:,4:7]


# In[ ]:


x


# In[ ]:





# In[ ]:


y=orvm_data.iloc[:,-1]


# In[ ]:


y


# In[ ]:


orvm_data


# In[ ]:


orvm_data['Vendor'].unique()


# In[ ]:


orvm_data.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import metrics


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)



modeld=DecisionTreeClassifier(criterion='gini', random_state=100,max_leaf_nodes=15)
modeld.fit(x_train,y_train)
prediction=modeld.predict(x_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test)*100)


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import graphviz 
import matplotlib.pyplot as plt


# In[ ]:


tree.plot_tree(modeld)


# In[ ]:


fn=['Price','Lead Time','Quality']
cn=['Forvia','Ficosa Internacional SA','Magna International Inc.','Mitsuba Corp','Murakami Corporation','Murakami Corporation','A1','ABC','XYZ','G Corporation','RK','Suppliers']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,30), dpi=200)
tree.plot_tree(modeld,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:





# In[ ]:


def prediction(x_test, modeld):
  
    # Predicton on test with giniIndex
    prediction1=modeld.predict(x_test)
    print("Predicted values:")
    print(prediction1)
    return prediction1


# In[ ]:


prediction(x_test, modeld)

# Function to calculate accuracy
def cal_accuracy(y_test, prediction1):
    
    print("Report : ",
    classification_report(y_test, prediction1))



cal_accuracy(y_test,prediction1)









