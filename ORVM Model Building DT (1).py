#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st


# In[ ]:


orvm_data=pd.read_csv("C:/Users/sunaina.s.ram/Downloads/ORVM Dataset.csv")


# In[ ]:


orvm_data


# In[ ]:


orvm_data.info()


# In[ ]:


orvm_data.head()


# In[ ]:


orvm_data=orvm_data.dropna()
orvm_data


# In[ ]:


# Import label encoder
from sklearn.preprocessing import LabelEncoder

# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

# Encode labels in column 'species'.
orvm_data['Vendor']= label_encoder.fit_transform(orvm_data['Vendor'])


# In[ ]:


orvm_data['Vendor'].unique()


# In[ ]:


orvm_data['Vehicle Type ']= label_encoder.fit_transform(orvm_data['Vehicle Type '])
orvm_data['Vehicle Type ']


# In[ ]:


orvm_data['ORVM Type']=label_encoder.fit_transform(orvm_data['ORVM Type'])


# In[ ]:


orvm_data['ORVM Type']


# In[ ]:


orvm_data['Auto Dimming ']=label_encoder.fit_transform(orvm_data['Auto Dimming '])
orvm_data['Auto Dimming ']


# In[ ]:


orvm_data.dtypes


# In[ ]:


orvm_data['Vendor'].unique()


# In[ ]:


orvm_data['Vendor'].value_counts()


# In[ ]:


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





# In[ ]:


y.to_csv("Vendor codes2.csv")


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


# model = svm.SVC() #select the algorithm
# model.fit(x_train,y_train) # we train the algorithm with the training data and the training output
# prediction=model.predict(x_test) #now we pass the testing data to the trained algorithm
# print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))#now we check the accuracy of the algorithm. 
# #we pass the predicted output by the model and the actual output

# model = LogisticRegression()
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test))

# In[ ]:


modeld=DecisionTreeClassifier(criterion='gini', random_state=100)
modeld.fit(x_train,y_train)
predictionn=modeld.predict(x_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(predictionn,y_test)*100)


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
cn=['A1','ABC','XYZ','Forvia','G Corporation','RK','Suppliers','M Suppliers','Ficosa Internacional SA','Magna International Inc.','Mitsuba Corp']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,30), dpi=300)
tree.plot_tree(modeld,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:


def prediction2(x_test, modeld):
  
    # Predicton on test with giniIndex
    prediction1=modeld.predict(x_test)
    print("Predicted values:")
    print(prediction1)
    return prediction1


# In[ ]:


prediction2(x_test, modeld)


# In[ ]:


# Function to calculate accuracy
def cal_accuracy(y_test, prediction):
    
    print("Report : ",)
    print(classification_report(y_test,predictionn))    


# In[ ]:


cal_accuracy(y_test, predictionn)


# In[ ]:


# selecting rows based on condition 
rslt_df = orvm_data[(orvm_data['Lead Time'] <= 3)]  
rslt_df


# In[ ]:


rslt_df1=rslt_df[(rslt_df['Quality']==2)]  
rslt_df1


# In[ ]:


rslt_df1.tail()


# ### Vendor 'M SUPPLIERS'  is selected for least lead time and high Quality ORVM 

# In[ ]:


get_ipython().run_line_magic('pip', 'freeze > requirements.txt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


min(orvm_data['Lead Time']) and max(orvm_data['Quality'])


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf1=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf1.fit(x_train,y_train)

y_pred1=clf1.predict(x_test)


# In[ ]:


print('The accuracy of the Random Forest is',metrics.accuracy_score(y_pred1,y_test)*100)


# In[ ]:





# In[ ]:


confusion_matrix(y_test,predictionn)


# In[ ]:


confusion_matrix(y_test,y_pred1)


# In[ ]:


pip install streamlit


# In[ ]:


import streamlit


# In[ ]:





# In[ ]:




