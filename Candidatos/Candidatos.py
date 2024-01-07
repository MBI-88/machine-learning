#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
np.random.seed(42)

dato=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/primary_results.csv')
dato.head()


# In[99]:


dato.iloc[200]


# In[100]:


variable_obj='party'
variable_numericas=dato.drop(variable_obj,axis=1).select_dtypes(np.number).columns
variable_categoricas=dato.drop('party',axis=1).select_dtypes(np.object).columns
variable_categoricas


# In[101]:


from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score,RandomizedSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.decomposition import PCA,TruncatedSVD

class ColumExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,columns,output_type='matrix'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        if self.output_type=='matrix':
            return X[self.columns].values
        
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser matrix o dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self
    
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def  rmse_cv(estimador,X,y):
    preds=estimador.predict(X)
    return rmse(y,preds)

dato[variable_obj]=dato['party'].replace({'Republican':0,'Democrat':1})


# In[102]:


pipeline_categorico=Pipeline([
    ('selector',ColumExtractor(columns=variable_categoricas,output_type='dataframe')),
    ('categoricas',OneHotEncoder()),
    ('imputador',SimpleImputer(strategy='constant',fill_value=0)),
    ('escalador',MinMaxScaler())
])
pipeline_numerico=Pipeline([
    ('selector',ColumExtractor(columns=variable_numericas,output_type='dataframe')),
    ('imputador',SimpleImputer(strategy='constant',fill_value=0)),
    ('escalador',MinMaxScaler())
])
pipeline_union=FeatureUnion([
    ['pipeline_cate',pipeline_categorico],
    ['pipeline_nume',pipeline_numerico],
])
pipeline_ada=Pipeline([
     ('union',pipeline_union),
    ('reductor',PCA()),
    ('estimador',AdaBoostClassifier())
])


# In[103]:


from scipy.stats import randint
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore',category=DataConversionWarning)

parametros={
    'estimador__algorithm':['SAMME','SAMME.R'] ,
    'estimador__base_estimator':[None] ,
    'estimador__learning_rate':np.logspace(-1,0.0), 
    'estimador__n_estimators':np.linspace(10,100,10).astype(int) , 
    'estimador__random_state': randint(13,50),
    'reductor':[PCA(),TruncatedSVD()],
    'reductor__n_components':randint(10,50),
    'reductor__random_state':np.linspace(10,100,10).astype(int),
    
}


# In[104]:


pipeline_estimador_Ada=Pipeline([
     ('union',pipeline_union),
    ('reductor',TruncatedSVD(algorithm='randomized',n_components=13,n_iter=5,random_state=60,tol=0.0)),
    ('estimador',AdaBoostClassifier(algorithm='SAMME.R',learning_rate=0.11513953993264472,n_estimators=20,random_state=18,base_estimator=None,))
])
pipeline_estimador_Ada.fit(dato,dato[variable_obj])


# In[51]:


error


# In[105]:


from sklearn.externals import joblib
import json

joblib.dump(pipeline_estimador_Ada,'estimador_candidatos.pkl')

with open('columnas_candidatos.json','w') as fname:
    columnas_candidatos=dato.columns.to_list()
    json.dump(columnas_candidatos,fname)
    fname.close()

variables_dtype=dato.dtypes
variables_dtype={col:dato[col].dtype for col in dato.columns }
joblib.dump(variables_dtype,'candidatos_dtypes.pkl')


#   ***Prueba***

# In[106]:


def dict_to_df(diccionario,columnas,dtypes):
    df=pd.DataFrame([diccionario])
    for col,dtype in dtypes.items():
        if col in df.columns:
            df[col]=df[col].astype(dtype)
        else:
            df[col]=None
    return df
def candidato(array):
    valor=array[0]
    if valor==1:
        print('El candicato pertenece al partido Democrata')
    else:
        print('El candidato pertenece al partido Republicano')

obs=dato.to_dict(orient='record')[200]


# In[107]:


obs


# In[108]:


import json as js
import joblib as jo
from sklearn.base import TransformerMixin,BaseEstimator

class ColumExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,columns,output_type='matrix'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        if self.output_type=='matrix':
            return X[self.columns].values
        
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser matrix o dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self

with open('columnas_candidatos.json','r') as fname:
    columnas=js.load(fname)
    fname.close()
estimador=jo.load('estimador_candidatos.pkl')
dtypes=jo.load('candidatos_dtypes.pkl')


# In[109]:


valor=dict_to_df(obs,columnas,dtypes)


# In[110]:


entrada=estimador.predict(valor)
candidato(entrada)


# In[ ]:




