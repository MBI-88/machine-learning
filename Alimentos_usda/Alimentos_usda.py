#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/alimentos_usda.csv')
datos.shape


# In[2]:


datos.head()


# In[3]:


datos.columns


# In[3]:


variable_obj='Protein_(g)'
datos.isnull().sum()


# In[9]:


from sklearn.model_selection import cross_val_score,RandomizedSearchCV,learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import json5


# In[10]:


class ColumExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,columns,output_type='dataframe'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self

resultados={}
def metrica(estimador,X,y):
    predicho=estimador.predict(X)
    return np.sqrt(mean_squared_error(y,predicho))
def evaluacion(estimador,X,y):
    return cross_val_score(estimador,X,y,scoring=metrica,cv=10,n_jobs=-1).mean()


# In[11]:


col_num=datos.drop(variable_obj,axis=1).select_dtypes([np.number]).columns
col_cat=datos.drop(variable_obj,axis=1).select_dtypes([np.object]).columns


# In[12]:


pipeline_num=Pipeline([
    ('selector',ColumExtractor(columns=col_num)),
    ('imputador',SimpleImputer()),
    ('estandarizador',MinMaxScaler())
])
pipeline_cat=Pipeline([
    ('selector',ColumExtractor(columns=col_cat)),
    ('codificador',OneHotEncoder()),
    ('estandarizador',MinMaxScaler())
])
pipeline_procesado=FeatureUnion([
    ('pipeline_num',pipeline_num),
    ('pipeline_cat',pipeline_cat)
])


# In[13]:


pipeline_Elastic=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador',ElasticNet())
])
pipeline_Knn=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',KNeighborsRegressor(n_jobs=-1)),
])


# In[14]:


resultados['estimador_Elastic']=evaluacion(pipeline_Elastic,datos,datos[variable_obj])


# In[15]:


resultados['estimador_Knn']=evaluacion(pipeline_Knn,datos,datos[variable_obj])


# In[16]:


resultados


# In[ ]:


pipeline_Knn.get_params()


# In[17]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,49).astype(int),
 'reductor_dim__random_state':np.linspace(10,200).astype(int),
 'estimador__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'estimador__leaf_size': np.linspace(30,160).astype(int),
 'estimador__n_neighbors': np.linspace(1,20).astype(int),
 'estimador__p': [1,2],
 'estimador__weights': ['uniform','distance']
}


# In[19]:


random_search=RandomizedSearchCV(estimator=pipeline_Knn,param_distributions=busqueda,scoring=metrica,cv=10,random_state=42,n_iter=10,n_jobs=-1)
random_search.fit(datos,datos[variable_obj])


# In[20]:


random_search.best_estimator_.steps


# In[21]:


pipeline_Knn_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=1, random_state=48,
      svd_solver='auto', tol=0.0, whiten=False)),
    ('estimador',KNeighborsRegressor(algorithm='brute', leaf_size=138, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=2, p=1,
                      weights='uniform')),
])


# In[25]:


resultados['estimador_Knn_opti']=evaluacion(pipeline_Knn_opti,datos,datos[variable_obj])


# In[26]:


resultados


# In[36]:


plt.plot(datos['Energ_Kcal'],datos[variable_obj],marker='.',ls='-',lw=0.15,color='r')
plt.title('Relacion Energia vs Proteinas',color='w')
plt.xlabel('Proteinas',color='w')
plt.ylabel('Energia',color='w');


# In[4]:


datos[variable_obj].plot.hist()


# In[5]:


datos.plot.scatter(x='Energ_Kcal',y=variable_obj);


# In[6]:


datos.plot(x='Energ_Kcal',y=variable_obj);


# In[ ]:




