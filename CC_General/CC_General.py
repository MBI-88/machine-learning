# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:23:52 2020

@author: MBI
"""
" Seleccion del dataframe con variable objetivo y datos "

import  pandas as  pd
import numpy as np
np.random.seed(42)
datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/CC GENERAL.csv')

variable_obj='PURCHASES'
variable_datos=datos.drop(variable_obj,axis=1).select_dtypes(np.number).columns
datos[variable_obj]=datos[variable_obj].fillna(0)
#%%
" Seleccion de los modulos sklearn "

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate,RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint as sp_randint
import json
#%%

" Selector de columnas "

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

pipeline_selector=Pipeline([
    ('selector_columnas',ColumExtractor(variable_datos)),
    ('imputador',SimpleImputer()),  
    ])

#%%
" Estableciendo los estiimadores "

pipeline_estimador_SVR=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',PCA()),
    ('estimador_SVR',SVR())
    ])
pipeline_estimador_AdaBoost=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',PCA()),
    ('estimador_AdaBoost',AdaBoostRegressor())
    ])
pipeline_estimador_KNe=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',PCA()),
    ('estimador',KNeighborsRegressor())
    ])
#%%

diccionario_pruebas={}
evaluacion=cross_validate(pipeline_estimador_SVR,datos,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_SVR']=evaluacion
evaluacion=cross_validate(pipeline_estimador_AdaBoost,datos,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_AdBoost']=evaluacion
evaluacion=cross_validate(pipeline_estimador_KNe,datos,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_KNe']=evaluacion
#%%
" Optimizando Adaboot() "

print(pipeline_estimador_AdaBoost.fit(datos,datos[variable_obj]))
#%%
print(pipeline_estimador_AdaBoost.get_params())
#%%
busqueda_parametros={
    'dimensionador':[PCA(),TruncatedSVD()],
    'dimensionador__n_components': sp_randint(1,50), 
    'dimensionador__random_state': range(10,150), 
    'estimador_AdaBoost__n_estimators': sp_randint(50,250), 
    'estimador_AdaBoost__random_state': sp_randint(10,150)
    }
adaboost_randoms=RandomizedSearchCV(
        estimator=pipeline_estimador_AdaBoost,
        n_iter=10,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        param_distributions=busqueda_parametros,
        cv=10,
        random_state=42,
        return_train_score=True
    )
adaboost_randoms.fit(datos,datos[variable_obj])
print(adaboost_randoms.best_estimator_.steps)
#%%
pipeline_estimador_AdaBoost_Opt=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',TruncatedSVD(algorithm='randomized', n_components=3, n_iter=5, random_state=60,
             tol=0.0)),
    ('estimador_AdaBoost',AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                  n_estimators=184, random_state=30))
    ])
evaluacion=cross_validate(pipeline_estimador_AdaBoost_Opt,datos,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_adaboost_opt']=evaluacion
#%%
"  Optimizando KN() "

pipeline_estimador_KNe.fit(datos,datos[variable_obj])
print(pipeline_estimador_KNe.get_params())
#%%
busqueda_parametros={
    'dimensionador':[PCA(),TruncatedSVD()],
    'dimensionador__n_components': sp_randint(1,50), 
    'dimensionador__random_state': range(10,150), 
    'estimador__algorithm': ['auto','ball_tree', 'kd_tree', 'brute'], 
    'estimador__leaf_size': sp_randint(30,150), 
    'estimador__metric': ['minkowski','precomputed'],   
    'estimador__n_neighbors': sp_randint(5,150), 
    'estimador__p': sp_randint(2,10), 
    'estimador__weights': ['uniform','distance']
    }
kn_randoms=RandomizedSearchCV(
        estimator=pipeline_estimador_KNe,
        n_iter=10,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        param_distributions=busqueda_parametros,
        cv=10,
        random_state=42,
        return_train_score=True
    )
kn_randoms.fit(datos,datos[variable_obj])
print(kn_randoms.best_estimator_.steps)
#%%
pipeline_estimador_KNe_Opt=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',TruncatedSVD(algorithm='randomized', n_components=3, n_iter=5, random_state=60,
             tol=0.0)),
    ('estimador',KNeighborsRegressor(algorithm='kd_tree', leaf_size=50, metric='minkowski',
                    metric_params=None, n_jobs=-1, n_neighbors=22, p=5,
                    weights='uniform'))
    
    ])
evaluacion=cross_validate(pipeline_estimador_KNe_Opt,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_kn_opt']=evaluacion
#%%
" Optimizando SVR() "

pipeline_estimador_SVR.fit(datos,datos[variable_obj])
#%%
print(pipeline_estimador_SVR.get_params())
#%%
busqueda_parametros={
    'dimensionador':[PCA(),TruncatedSVD()],
    'dimensionador__n_components': sp_randint(1,50), 
    'dimensionador__random_state': range(10,150), 
    'estimador_SVR__degree': sp_randint(3,15), 
    'estimador_SVR__gamma': ['scale','auto'],  
    'estimador_SVR__shrinking': [True,False], 
    }
svr_randoms=RandomizedSearchCV(
        estimator=pipeline_estimador_SVR,
        n_iter=10,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        param_distributions=busqueda_parametros,
        cv=10,
        random_state=42,
        return_train_score=True
    )
svr_randoms.fit(datos,datos[variable_obj])
#%%
print(svr_randoms.best_estimator_.steps)
pipeline_estimador_SVR_Opt=Pipeline([
    ('pipeline_selector',pipeline_selector),
    ('dimensionador',TruncatedSVD(algorithm='randomized', n_components=3, n_iter=5, random_state=60,
             tol=0.0)),
    ('estimador_SVR',SVR(C=1.0, cache_size=200, coef0=0.0, degree=9, epsilon=0.1, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))
    ])
evaluacion=cross_validate(pipeline_estimador_SVR_Opt,datos[variable_obj],scoring='neg_mean_absolute_error',n_jobs=-1,cv=10)
diccionario_pruebas['estimador_svr_opt']=evaluacion
#%%
"Eleccion del modelo para el resultado final"

print(diccionario_pruebas)
#%%
" Ensamblado final "

salida=pipeline_estimador_KNe_Opt.fit(datos,datos[variable_obj]).predict(datos)
print(datos[variable_obj][:10])
print('\n')
print(salida[:10])
#%%
" Exportando modelo "     

with open('Columnas_CC_General.json','w') as fname:
    columnas=datos.columns.tolist()
    json.dump(columnas,fname)
    fname.close()


cc_general_dtypes=datos.dtypes
cc_general_dtypes={col:datos[col].dtype for col in datos.columns}
joblib.dump(cc_general_dtypes,'TiposDatos_CC_General.pkl')

joblib.dump(pipeline_estimador_KNe_Opt,'CC_General.pkl')


    
                               








