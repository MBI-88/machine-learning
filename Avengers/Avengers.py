# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:48:56 2020

@author: MBI
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import learning_curve
from scipy.sparse import issparse
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV,space
from sklearn.decomposition import PCA

np.random.seed(42)

class Transformador_Base(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X

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
        self.is_fitted = True
        return self
    
    def fit_transform(self,X,y=None):
        return self.transform(X=X,y=y)

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns, output_type="dataframe"):
        self.columns = columns
        self.output_type = output_type

    def transform(self, X, **transform_params):
        if isinstance(X, list):
            X = pd.DataFrame.from_dict(X)
            
        elif self.output_type == "dataframe":
            return X[self.columns]
        raise Exception("output_type tiene que ser matrix o dataframe")
        
    def fit(self, X, y=None, **fit_params):
        return self
        
    def fit(self, X, y=None, **fit_params):
        return self

# Main()

datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/avengers.csv').drop('URL',axis=1)

variable_obj='n_apariciones'

col_num=datos.drop(variable_obj,axis=1).select_dtypes(np.number).columns
col_categoricas=datos.drop('Notes',axis=1).select_dtypes([np.object]).columns
col_text='Notes'
datos[col_text]=datos[col_text].fillna('Null')

pipeline_num=Pipeline([
    ['extractor',ColumnExtractor(columns=col_num,output_type='dataframe')],
    ['imputador',SimpleImputer()],
    ['escalador',StandardScaler()]])

pipeline_categorico=Pipeline([
    ['extractor',ColumnExtractor(columns=col_categoricas,output_type='dataframe')],
    ['codificador',OneHotEncoder()],
    ['escalador',StandardScaler()],
    ])


pipeline_text=Pipeline([
    ['extractor',ColumnExtractor(columns=col_text,output_type='dataframe')],
    ['vectorizador',TfidfVectorizer()],
    ['transformador_denso',DenseTransformer()],
    ])
#%%
resultados={}
def rmse(estimador,X,y):
    predic=estimador.predict(X)
    return np.sqrt(mean_squared_error(y,predic))
    
def evaluar_modelo(estimador,X,y):
    resultados_estimador=cross_validate(estimador,X,y,n_jobs=-1,cv=10,return_train_score=True,scoring=rmse)
    return resultados_estimador

def ver_resultados():
    resultados_df=pd.DataFrame(resultados).T
    for col in resultados_df:
        resultados_df[col]=resultados_df[col].apply(np.mean)
        resultados_df[col+'_idx']=resultados_df[col]/resultados_df[col].max()
    return print(resultados_df)

pipeline_procesado=FeatureUnion([
    ['pipeline numerico',pipeline_num],
    ['pipeline_categorico',pipeline_categorico],
    ['pipeline_texto',pipeline_text]],n_jobs=-1)

pipeline_estimador_random=Pipeline([
    ['pipeline_procesado',pipeline_procesado],
    ['dimensionalidad',PCA(random_state=42,n_components=2)],
    ['estimador',RandomForestRegressor(n_jobs=-1,random_state=42,n_estimators=81)]
    ])

resultados['estimador_random']=evaluar_modelo(pipeline_estimador_random,datos,datos[variable_obj])

valor=pipeline_estimador_random.fit(datos,datos[variable_obj]).predict((datos))
prueba=mean_absolute_error(datos[variable_obj],valor).mean()
print(prueba)
print(valor[:20])
print(datos[variable_obj][:20])

#%%
busqueda_skop={
    'estimador__min_samples_split': space.Real(0.001,0.99),
    'estimador__min_samples_leaf': space.Real(0.001,0.5),
    'estimador__max_samples': space.Integer(10,100),
    'estimador__max_depth': space.Integer(3,10),
    'estimador__n_estimators':space.Integer(10,1000), 
    'estimador__random_state':space.Integer(10,120),
    'dimensionalidad__random_state':space.Integer(10,80),

    }
skop_selector=BayesSearchCV(
    estimator=pipeline_estimador_random,
    search_spaces=busqueda_skop,
    n_iter=100,
    scoring=rmse,
    n_jobs=-1,
    cv=10,
    random_state=42
    )
skop_selector.fit(datos,datos[variable_obj])
#%%
print(skop_selector.best_estimator_.steps)
#%%

pipeline_estimador_RandomForest=Pipeline([
    ['pipeline_procesado',pipeline_procesado],
    ['dimensionalidad',PCA(random_state=10)],
    ['estimador',RandomForestRegressor(max_depth=9, max_features='auto', max_leaf_nodes=None, max_samples=100,min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=0.001,
                      min_samples_split=0.001, min_weight_fraction_leaf=0.0,
                      n_estimators=10, n_jobs=-1, oob_score=False,
                      random_state=120, verbose=0, warm_start=False
        
        )]
    ])

prueba=pipeline_estimador_RandomForest.fit(datos,datos[variable_obj]).predict(datos)
valor=mean_absolute_error(datos[variable_obj],pipeline_estimador_RandomForest.fit(datos,datos[variable_obj]).predict((datos))).mean()
print(valor)
print(prueba[:5])
print(datos[variable_obj][:5])
#%%
import matplotlib.pyplot as plt


np.linspace(0.001,1.,10)
train_size,train_scores,test_scores=learning_curve(
    pipeline_estimador_random,datos,datos[variable_obj],cv=5,n_jobs=-1,scoring='neg_mean_squared_error',train_sizes=np.linspace(0.001,1.,10))


train_scores_mean=np.mean(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)

plt.plot(train_size,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamineto')
plt.plot(train_size,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curva de Apendizaje: Random_Forest_81_estimadores')
plt.xlabel('Numero de  muestras de entrenamiento')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();

#%%
import json
joblib.dump(pipeline_estimador_random,'pipeline_avengers.pkl')
#%%
with open('columnas_avengers.json','w') as fname:
    columnas_avengers=datos.columns.tolist()
    json.dump(columnas_avengers,fname)
    fname.close()
#%%
datos_dt=datos.dtypes
datos_dt={col:datos[col].dtype for col in datos.columns}
joblib.dump(datos_dt,'datos_avengers.pkl')
#%%
# Prueba de una observacion

nueva_obs=datos.to_dict(orient='record')[10]
print(nueva_obs)
#%%
dic={
  'nombre': 'Jacques Duquesne', 
  'n_apariciones': 115, 
  'actual': 'NO', 
  'genero': 'MALE', 
  'fecha_inicio': 1965, 
  'Notes': 'Dies in Avengers_Vol_1_130. Brought back by the Chaos King'
     }

with  open('columnas_avengers.json','r') as fname:
    columnas_avengers=json.load(fname)
    fname.close()
datos_avengers=joblib.load('datos_avengers.pkl')
load=joblib.load('pipeline_avengers.pkl')
def dict_a_df(obs,columnas,dtypes):
    obs_df=pd.DataFrame([obs])
    for col,dtype in dtypes.items():
        if col in obs_df.columns:
            obs_df[col]=obs_df[col].astype(dtype)
        else:
            obs_df[col]=None
    return obs_df

observaion=dict_a_df(dic,columnas_avengers,datos_avengers)
#%%
valor=load.predict(observaion)
print('Numero de apariciones del personaje {} '.format(valor))




