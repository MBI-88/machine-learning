#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(42)

data=pd.read_csv('noticias_meneame_short.csv')
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data['Categoria']=data['Categoria'].fillna('Desconocido')
data['Descripcion']=data['Descripcion'].fillna('None')


# In[17]:


from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from  sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import json5


# In[6]:


with open('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/stopwords-es.json')as fname:
    stopwords_es= json5.load(fname)
    fname.close()
    
class DenseTransformer(BaseEstimator):
    def __init__(self,return_copy=True):
        self.return_copy=return_copy
        self.is_fitted=False
    def transform(self,X,y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X
    def fit(self,X,y=None):
        self.is_fitted=True
        return self
    def fit_transform(self,X,y=None):
        return self.transform(X=X,y=y)


resultados={}
def puntuacion(estimador,X,y):
    predic=estimador.predict(X)
    return f1_score(y,predic,average='micro')
def evaluar_modelo(estimador,X,y):
    resultados_estimador=cross_validate(estimador,X,y,n_jobs=-1,cv=10,return_train_score=True,scoring=puntuacion)
    return resultados_estimador

def ver_resultados():
    resultados_df=pd.DataFrame(resultados).T
    return print(resultados_df)


# In[7]:


pipeline_procesado=Pipeline([
    ('vectorizador',TfidfVectorizer(strip_accents='unicode',stop_words=stopwords_es,max_features=10000)),
    ('transformador',DenseTransformer()),
    ('estandarizador',MaxAbsScaler())
])
pipeline_procesado_B=Pipeline([
    ('vectorizador',CountVectorizer(strip_accents='unicode',stop_words=stopwords_es,binary=True,max_features=10000)),
    ('transformador',DenseTransformer()),
    ('estandarizador',MaxAbsScaler())
])


# In[8]:


pipeline_Gaussiano=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador_Gaussiano',GaussianNB())
])
pipeline_Multinominal=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador_Multi',MultinomialNB())
])
pipeline_Bernoli=Pipeline([
    ('pipeline_procesado_B',pipeline_procesado_B),
    ('estimador_Bernoli',BernoulliNB())
])


# In[9]:


resultados['estimador_Gaussiano']=evaluar_modelo(pipeline_Gaussiano,data.Descripcion,data.Categoria)
resultados['estiamdor_Multi']=evaluar_modelo(pipeline_Multinominal,data.Descripcion,data.Categoria)
resultados['estimador_Bernoli']=evaluar_modelo(pipeline_Bernoli,data.Descripcion,data.Categoria)


# In[10]:


ver_resultados()


# In[11]:


pipeline_Gaussiano.fit(data.Descripcion,data.Categoria)
pipeline_Gaussiano.score(data.Descripcion,data.Categoria)


# In[14]:


pipeline_Bernoli.fit(data.Descripcion,data.Categoria)
pipeline_Bernoli.score(data.Descripcion,data.Categoria)


# In[15]:


pipeline_Multinominal.fit(data.Descripcion,data.Categoria)
pipeline_Multinominal.score(data.Descripcion,data.Categoria)


# In[44]:


dtype_noticias=data.dtypes
dtype_noticias={col:data[col].dtypes for col in data.columns}
joblib.dump(dtype_noticias,'Noticias_meneame_dtypes.pkl')
joblib.dump(pipeline_Multinominal,'Noticias_meneame_estimator.pkl')

with open('Noticias_meneame_columns.json','w') as fname:
    col_noticias=data.columns.to_list()
    json5.dump(col_noticias,fname)
    fname.close()


# In[68]:


def dict_a_df(obs,columnas,dtypes):
    obs_df=pd.DataFrame([obs])
    for col,dtype in dtypes.items():
        if col in obs_df.columns:
            obs_df[col]=obs_df[col].astype(dtype)
        else:
            obs_df[col]=None
    return obs_df.Descripcion


# In[69]:


obs=data.to_dict(orient='record')[0]
obs


# In[70]:


obs=data.to_dict(orient='record')[19]
with open('Noticias_meneame_columns.json','r') as fname:
    columnas=json5.load(fname)
    fname.close()
dtype=joblib.load('Noticias_meneame_dtypes.pkl')
clf=joblib.load('Noticias_meneame_estimator.pkl')

obs_df=dict_a_df(obs,columnas,dtype)
obs_df


# In[71]:


clf.predict(obs_df)


# In[74]:


data[['Descripcion','Categoria']]


# In[ ]:




