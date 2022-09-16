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





# In[113]:


pip install streamlit


# In[114]:


import streamlit


# In[116]:



# pickling the model
import pickle
pickle_out = open("decsiontreeORVMclassifier.pkl", "wb")
pickle.dump(modeld, pickle_out)
pickle_out.close()


# In[ ]:





# In[120]:


get_ipython().run_line_magic('pip', 'freeze > requirements.txt')


# In[ ]:





# In[4]:


pip install streamlit


# In[3]:


pip list


# In[ ]:


aiohttp                       3.8.1
aiosignal                     1.2.0
alabaster                     0.7.12
altair                        4.2.0
anaconda-client               1.9.0
anaconda-navigator            2.1.4
anaconda-project              0.10.2
anyio                         3.5.0
appdirs                       1.4.4
argon2-cffi                   21.3.0
argon2-cffi-bindings          21.2.0
arrow                         1.2.2
astroid                       2.6.6
astropy                       5.0.4
asttokens                     2.0.5
async-timeout                 4.0.1
atomicwrites                  1.4.0
attrs                         21.4.0
Automat                       20.2.0
autopep8                      1.6.0
Babel                         2.9.1
backcall                      0.2.0
backports.functools-lru-cache 1.6.4
backports.tempfile            1.0
backports.weakref             1.0.post1
bcrypt                        3.2.0
beautifulsoup4                4.11.1
binaryornot                   0.4.4
bitarray                      2.4.1
bkcharts                      0.2
black                         19.10b0
bleach                        4.1.0
blinker                       1.5
bokeh                         2.4.2
boto3                         1.21.32
botocore                      1.24.32
Bottleneck                    1.3.4
brotlipy                      0.7.0
cachetools                    4.2.2
certifi                       2021.10.8
cffi                          1.15.0
chardet                       4.0.0
charset-normalizer            2.0.4
click                         8.0.4
cloudpickle                   2.0.0
clyent                        1.2.2
collection                    0.1.6
colorama                      0.4.4
colorcet                      2.0.6
commonmark                    0.9.1
comtypes                      1.1.10
conda                         4.13.0
conda-build                   3.21.8
conda-content-trust           0+unknown
conda-pack                    0.6.0
conda-package-handling        1.8.1
conda-repo-cli                1.0.4
conda-token                   0.3.0
conda-verify                  3.4.2
constantly                    15.1.0
cookiecutter                  1.7.3
cryptography                  3.4.8
cssselect                     1.1.0
cycler                        0.11.0
Cython                        0.29.28
cytoolz                       0.11.0
daal4py                       2021.5.0
dask                          2022.2.1
datashader                    0.13.0
datashape                     0.5.4
debugpy                       1.5.1
decorator                     5.1.1
defusedxml                    0.7.1
diff-match-patch              20200713
distributed                   2022.2.1
docutils                      0.17.1
entrypoints                   0.4
et-xmlfile                    1.1.0
Note: you may need to restart the kernel to use updated packages.executing                     0.8.3
fastjsonschema                2.15.1
filelock                      3.6.0
findspark                     2.0.1
flake8                        3.9.2
Flask                         1.1.2
fonttools                     4.25.0

frozenlist                    1.2.0
fsspec                        2022.2.0
future                        0.18.2
gensim                        4.1.2
gitdb                         4.0.9
GitPython                     3.1.27
glob2                         0.7
google-api-core               1.25.1
google-auth                   1.33.0
google-cloud-core             1.7.1
google-cloud-storage          1.31.0
google-crc32c                 1.1.2
google-resumable-media        1.3.1
googleapis-common-protos      1.53.0
graphviz                      0.20.1
greenlet                      1.1.1
grpcio                        1.42.0
h5py                          3.6.0
HeapDict                      1.0.1
holoviews                     1.14.8
hvplot                        0.7.3
hyperlink                     21.0.0
idna                          3.3
imagecodecs                   2021.8.26
imageio                       2.9.0
imagesize                     1.3.0
importlib-metadata            4.11.3
incremental                   21.3.0
inflection                    0.5.1
iniconfig                     1.1.1
intake                        0.6.5
intervaltree                  3.1.0
ipykernel                     6.9.1
ipython                       8.2.0
ipython-genutils              0.2.0
ipywidgets                    7.6.5
isort                         5.9.3
itemadapter                   0.3.0
itemloaders                   1.0.4
itsdangerous                  2.0.1
jdcal                         1.4.1
jedi                          0.18.1
Jinja2                        2.11.3
jinja2-time                   0.2.0
jmespath                      0.10.0
joblib                        1.1.0
json5                         0.9.6
jsonschema                    4.4.0
jupyter                       1.0.0
jupyter-client                6.1.12
jupyter-console               6.4.0
jupyter-core                  4.9.2
jupyter-server                1.13.5
jupyterlab                    3.3.2
jupyterlab-pygments           0.1.2
jupyterlab-server             2.10.3
jupyterlab-widgets            1.0.0
keyring                       23.4.0
kiwisolver                    1.3.2
lazy-object-proxy             1.6.0
libarchive-c                  2.9
llvmlite                      0.38.0
locket                        0.2.1
lxml                          4.8.0
Markdown                      3.3.4
MarkupSafe                    2.0.1
matplotlib                    3.5.1
matplotlib-inline             0.1.2
mccabe                        0.6.1
menuinst                      1.4.18
mistune                       0.8.4
mkl-fft                       1.3.1
mkl-random                    1.2.2
mkl-service                   2.4.0
mock                          4.0.3
mpmath                        1.2.1
msgpack                       1.0.2
multidict                     5.1.0
multipledispatch              0.6.0
munkres                       1.1.4
mypy-extensions               0.4.3
navigator-updater             0.2.1
nbclassic                     0.3.5
nbclient                      0.5.13
nbconvert                     6.4.4
nbformat                      5.3.0
nest-asyncio                  1.5.5
networkx                      2.7.1
nltk                          3.7
nose                          1.3.7
notebook                      6.4.8
numba                         0.55.1
numexpr                       2.8.1
numpy                         1.21.5
numpydoc                      1.2
olefile                       0.46
openpyxl                      3.0.9
packaging                     21.3
pandas                        1.4.2
pandocfilters                 1.5.0
panel                         0.13.0
param                         1.12.0
paramiko                      2.8.1
parsel                        1.6.0
parso                         0.8.3
partd                         1.2.0
pathspec                      0.7.0
patsy                         0.5.2
pep8                          1.7.1
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        9.0.1
pip                           21.2.4
pkginfo                       1.8.2
plotly                        5.6.0
pluggy                        1.0.0
poyo                          0.5.0
prometheus-client             0.13.1
prompt-toolkit                3.0.20
Protego                       0.1.16
protobuf                      3.19.1
psutil                        5.8.0
ptyprocess                    0.7.0
pure-eval                     0.2.2
py                            1.11.0
py4j                          0.10.9.3
pyarrow                       8.0.0
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycodestyle                   2.7.0
pycosat                       0.6.3
pycparser                     2.21
pyct                          0.4.6
pycurl                        7.44.1
pydeck                        0.8.0b3
PyDispatcher                  2.0.5
pydocstyle                    6.1.1
pydotplus                     2.0.2
pyerfa                        2.0.0
pyflakes                      2.3.1
Pygments                      2.11.2
PyHamcrest                    2.0.2
PyJWT                         2.1.0
pylint                        2.9.6
pyls-spyder                   0.4.0
Pympler                       1.0.1
PyNaCl                        1.4.0
pyodbc                        4.0.32
pyOpenSSL                     21.0.0
pyparsing                     3.0.4
pyreadline                    2.1
pyrsistent                    0.18.0
PySocks                       1.7.1
pyspark                       3.2.1
pytest                        7.1.1
python-dateutil               2.8.2
python-lsp-black              1.0.0
python-lsp-jsonrpc            1.0.0
python-lsp-server             1.2.4
python-slugify                5.0.2
python-snappy                 0.6.0
pytz                          2021.3
pytz-deprecation-shim         0.1.0.post0
pyviz-comms                   2.0.2
PyWavelets                    1.3.0
pywin32                       302
pywin32-ctypes                0.2.0
pywinpty                      2.0.2
PyYAML                        6.0
pyzmq                         22.3.0
QDarkStyle                    3.0.2
qstylizer                     0.1.10
QtAwesome                     1.0.3
qtconsole                     5.3.0
QtPy                          2.0.1
queuelib                      1.5.0
regex                         2022.3.15
requests                      2.27.1
requests-file                 1.5.1
rich                          12.5.1
rope                          0.22.0
rsa                           4.7.2
Rtree                         0.9.7
ruamel-yaml-conda             0.15.100
s3transfer                    0.5.0
scikit-image                  0.19.2
scikit-learn                  1.0.2
scikit-learn-intelex          2021.20220215.102710
scipy                         1.7.3
Scrapy                        2.6.1
seaborn                       0.11.2
semver                        2.13.0
Send2Trash                    1.8.0
service-identity              18.1.0
setuptools                    61.2.0
sip                           4.19.13
six                           1.16.0
smart-open                    5.1.0
smmap                         5.0.0
sniffio                       1.2.0
snowballstemmer               2.2.0
sortedcollections             2.1.0
sortedcontainers              2.4.0
soupsieve                     2.3.1
Sphinx                        4.4.0
sphinxcontrib-applehelp       1.0.2
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.0
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
spyder                        5.1.5
spyder-kernels                2.1.3
SQLAlchemy                    1.4.32
stack-data                    0.2.0
statsmodels                   0.13.2
streamlit                     1.12.2
sympy                         1.10.1
tables                        3.6.1
tabulate                      0.8.9
TBB                           0.2
tblib                         1.7.0
tenacity                      8.0.1
terminado                     0.13.1
testpath                      0.5.0
text-unidecode                1.3
textdistance                  4.2.1
threadpoolctl                 2.2.0
three-merge                   0.1.1
tifffile                      2021.7.2
tinycss                       0.4
tldextract                    3.2.0
toml                          0.10.2
tomli                         1.2.2
toolz                         0.11.2
tornado                       6.1
tqdm                          4.64.0
traitlets                     5.1.1
Twisted                       22.2.0
twisted-iocpsupport           1.0.2
typed-ast                     1.4.3
typing_extensions             4.1.1
tzdata                        2022.2
tzlocal                       4.2
ujson                         5.1.0
Unidecode                     1.2.0
urllib3                       1.26.9
validators                    0.20.0
w3lib                         1.21.0
watchdog                      2.1.6
wcwidth                       0.2.5
webencodings                  0.5.1
websocket-client              0.58.0
Werkzeug                      2.0.3
wheel                         0.37.1
widgetsnbextension            3.5.2
win-inet-pton                 1.1.0
win-unicode-console           0.5
wincertstore                  0.2
wrapt                         1.12.1
xarray                        0.20.1
xlrd                          2.0.1
XlsxWriter                    3.0.3
xlwings                       0.24.9
yapf                          0.31.0
yarl                          1.6.3
zict                          2.0.0
zipp                          3.7.0
zope.interface                5.4.0

