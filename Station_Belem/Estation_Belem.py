#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import RandomizedSearchCV,cross_validate
from sklearn.svm import SVR,LinearSVR
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from xgboost import XGBRegressor
from xgboost import plot_importance,to_graphviz
from sklearn.decomposition import PCA,TruncatedSVD
from scipy.sparse import issparse
import json5
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[12,12]
np.random.seed(48)


# In[2]:


datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/station_belem.csv')
datos.head()


# In[3]:


variable_obje='metANN'
varible_nume=datos.drop(variable_obje,axis=1).select_dtypes([np.number]).columns


# In[4]:


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns, output_type="dataframe"):
        self.columns = columns
        self.output_type = output_type

    def transform(self, X, **transform_params):
        if isinstance(X, list):
            X = pd.DataFrame.from_dict(X)
            
        elif self.output_type == "dataframe":
            return X[self.columns]
        raise Exception("output_type tiene que ser dataframe")
        
    def fit(self, X, y=None, **fit_params):
        return self
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[5]:


resultados={}
def valor_absoluto_medio(estimador,X,y):
    estimator=estimador.predict(X)
    return mean_absolute_error(y,estimator)

def evaluar_modelo(estimador,X,y):
    resultados_estimador=cross_validate(estimador,X,y,n_jobs=-1,cv=10,return_train_score=True,scoring=valor_absoluto_medio)
    return resultados_estimador

def ver_resultados():
    resultados_df=pd.DataFrame(resultados).T
    return print(resultados_df)


# In[6]:


pipeline_numerico=Pipeline([
    ('selector_columnas',ColumnExtractor(columns=varible_nume)),
    ('imputador',SimpleImputer()),
    ('estandarizador',MaxAbsScaler())
])
pipeline_procesado=FeatureUnion([
    ('pipeline_numerico',pipeline_numerico)
])


# In[7]:


pipeline_SVR=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_svr',SVR())
])
pipeline_LinealSVR=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_LinealSVR',LinearSVR())
])
pipeline_AdaBoos=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_AdaBoos',AdaBoostRegressor())
])
pipeline_ExtraTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_ExtraTree',ExtraTreesRegressor(n_jobs=-1))
])
pipeline_Gradiente=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_Gradinete',GradientBoostingRegressor())
])
pipeline_XGB=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador_XGB',XGBRegressor(n_jobs=-1))
])


# In[8]:


resultados['estimador_SVR']=evaluar_modelo(pipeline_SVR,datos,datos[variable_obje])
resultados['estimador_LinealSVR']=evaluar_modelo(pipeline_LinealSVR,datos,datos[variable_obje])
resultados['estimador_AdaBoos']=evaluar_modelo(pipeline_AdaBoos,datos,datos[variable_obje])
resultados['estiamdor_ExtraTree']=evaluar_modelo(pipeline_ExtraTree,datos,datos[variable_obje])
resultados['estiamador_XGB']=evaluar_modelo(pipeline_XGB,datos,datos[variable_obje])
resultados['estiamador_Gradiente']=evaluar_modelo(pipeline_Gradiente,datos,datos[variable_obje])


# In[9]:


ver_resultados()


# In[11]:


pipeline_SVR.fit(datos,datos[variable_obje])
pipeline_SVR.score(datos,datos[variable_obje])


# In[12]:


pipeline_SVR.get_params()


# In[13]:


busqueda_iperparametros={
 'estimador_svr__C': [1,10],
 'estimador_svr__kernel': ('rbf','linear','auto'),
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components':np.linspace(1,10).astype(int),
 'reductor_dim__random_state': np.linspace(10,100).astype(int),
}


# In[14]:


grid=RandomizedSearchCV(pipeline_SVR,param_distributions=busqueda_iperparametros,n_jobs=-1,scoring=valor_absoluto_medio,cv=10,n_iter=10,random_state=42)
grid.fit(datos,datos[variable_obje])


# In[15]:


grid.best_estimator_.steps


# In[16]:


pipeline_SVR_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=4,
                     random_state=28, svd_solver='auto', tol=0.0,
                     whiten=False)),
    ('estimador_svr',SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                     gamma='scale', kernel='rbf', max_iter=-1,
                     shrinking=True, tol=0.001, verbose=False))
])


# In[17]:


resultados['estimador_svr_opti']=evaluar_modelo(pipeline_SVR_opti,datos,datos[variable_obje])


# In[18]:


ver_resultados()


# In[36]:


predicho=pipeline_SVR_opti.fit(datos,datos[variable_obje]).predict(datos)
pipeline_SVR_opti.score(datos,datos[variable_obje])


# In[20]:


pipeline_XGB.fit(datos,datos[variable_obje])
pipeline_XGB.score(datos,datos[variable_obje])


# In[21]:


pipeline_XGB.get_params()


# In[24]:


busqueda_iperparametros={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components':np.linspace(1,10).astype(int),
 'reductor_dim__random_state': np.linspace(10,100).astype(int),
 'estimador_XGB__booster': ['gbtree','gblinear',' dart'],
 'estimador_XGB__importance_type': ["gain","weight", "cover", "total_gain","total_cover"],
 'estimador_XGB__max_depth': np.linspace(3,10).astype(int),
 'estimador_XGB__n_estimators': np.linspace(50,150,10).astype(int),
 'estimador_XGB__random_state': np.linspace(10,100).astype(int),
}


# In[25]:


grid=RandomizedSearchCV(pipeline_XGB,param_distributions=busqueda_iperparametros,cv=10,n_jobs=-1,scoring=valor_absoluto_medio,n_iter=10,random_state=42)
grid.fit(datos,datos[variable_obje])


# In[27]:


grid.best_estimator_.steps


# In[28]:


pipeline_XGB_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD(algorithm='randomized', n_components=10, n_iter=5, random_state=10,tol=0.0)),
    ('estimador_XGB',XGBRegressor(base_score=0.5, booster='gblinear', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=1, gamma=0,
               importance_type='gain', learning_rate=0.1, max_delta_step=0,
               max_depth=5, min_child_weight=1, missing=None, n_estimators=50,
               n_jobs=-1, nthread=None, objective='reg:linear', random_state=43,
               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
               silent=None, subsample=1, verbosity=1))
])


# In[31]:


resultados['estimador_XGB_opti']=evaluar_modelo(pipeline_XGB_opti,datos,datos[variable_obje])


# In[32]:


ver_resultados()


# In[33]:


pipeline_XGB_opti.fit(datos,datos[variable_obje])
pipeline_XGB_opti.score(datos,datos[variable_obje])


# **Ensamblado del modelo**

# In[34]:


with open('Station_columns.json','w') as fname:
    columns=datos.columns.to_list()
    json5.dump(columns,fname)
    fname.close()

station_dtypes=datos.dtypes
station_dtypes={col:datos[col].dtypes for col in datos.columns}
joblib.dump(station_dtypes,'Station_dtype.pkl')
joblib.dump(pipeline_XGB_opti,'Station_estimator.pkl')


# **Visualizando datos**

# In[37]:


plt.plot(datos[variable_obje],predicho,marker='.',ls='-',lw=0.40,color='r')
plt.title('Relacion entre valor real y predicho (estimador SVR optimizado)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho');


# In[38]:


predicho_0=pipeline_XGB_opti.predict(datos)


# In[39]:


plt.plot(datos[variable_obje],predicho_0,marker='.',ls='-',lw=0.40,color='r')
plt.title('Relacion entre valor real y predicho (estimador SVR)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho');

