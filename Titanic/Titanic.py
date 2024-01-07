#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import  matplotlib.pyplot   as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)

datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic.csv')
datos.head()


# In[2]:


datos.shape


# In[3]:


datos['puerto_salida'].unique()


# In[26]:


datos['puerto_salida']=datos.puerto_salida.replace({'S':1,'C':2,'Q':3})


# In[27]:


datos['genero'].unique()


# In[28]:


datos['genero']=datos.genero.replace({'hombre':1,'mujer':2})


# In[29]:


objetivo='superviviente'
col_num=datos.drop(objetivo,axis=1).select_dtypes([np.number]).columns


# In[2]:


from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_validate
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.externals import joblib
import json5
from sklearn.pipeline import Pipeline,FeatureUnion


# In[5]:


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
    
resultados={}
def evalucion(estimador,dato,objetivo):
    return cross_validate(estimador,dato,objetivo,scoring='roc_auc',cv=15,n_jobs=-1,return_train_score=True)
def ver_resultado():
    dataframe=pd.DataFrame(resultados).T
    return print(dataframe)


# In[11]:


pipeline_numerico=Pipeline([
    ('selector',ColumnExtractor(columns=col_num)),
    ('imputador',SimpleImputer()),
    ('estandarizador',MinMaxScaler())
])
pipeline_procesado=FeatureUnion([('pipeline_numerico',pipeline_numerico)])


# In[12]:


pipeline_DecisionTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',DecisionTreeClassifier())
])
pipeline_ExtraTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',ExtraTreeClassifier())
])
pipeline_Knn=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',KNeighborsClassifier(n_jobs=-1))
])
pipeline_AdaBoost=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',AdaBoostClassifier())
])
pipeline_Bagging=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',BaggingClassifier(n_jobs=-1))
])


# In[13]:


resultados['estimador_DecisionTree']=evalucion(pipeline_DecisionTree,datos,datos[objetivo])
resultados['estimador_ExtraTree']=evalucion(pipeline_ExtraTree,datos,datos[objetivo])
resultados['estimador_Knn']=evalucion(pipeline_Knn,datos,datos[objetivo])
resultados['estimador_AdaBoost']=evalucion(pipeline_AdaBoost,datos,datos[objetivo])
resultados['estimador_Bagging']=evalucion(pipeline_Bagging,datos,datos[objetivo])


# In[14]:


ver_resultado()


# In[15]:


pipeline_AdaBoost.fit(datos,datos[objetivo])
pipeline_AdaBoost.score(datos,datos[objetivo])


# In[16]:


pipeline_AdaBoost.get_params()


# In[17]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(0,7).astype(int),
 'reductor_dim__random_state': np.linspace(1,100).astype(int),
 'estimador__algorithm': ['SAMME.R','SAMME'],
 'estimador__learning_rate': np.linspace(0.01,1.0).astype(float),
 'estimador__n_estimators':np.linspace(50,100).astype(int),
 'estimador__random_state': np.linspace(10,100).astype(int)
}


# In[18]:


grid=RandomizedSearchCV(estimator=pipeline_AdaBoost,param_distributions=busqueda,scoring='roc_auc',cv=10,n_jobs=-1,random_state=42)
grid.fit(datos,datos[objetivo])


# In[19]:


grid.best_estimator_.steps


# In[20]:


pipeline_AdaBoost_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=6,
                     random_state=21, svd_solver='auto', tol=0.0,
                     whiten=False)),
    ('estimador',AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                    learning_rate=0.39387755102040817,
                                    n_estimators=97, random_state=88))
])


# In[21]:


pipeline_Bagging.fit(datos,datos[objetivo])
pipeline_Bagging.score(datos,datos[objetivo])


# In[22]:


pipeline_Bagging.get_params()


# In[23]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(0,7).astype(int),
 'reductor_dim__random_state': np.linspace(1,100).astype(int),
 'estimador__bootstrap': [True,False],
 'estimador__n_estimators': np.linspace(10,100).astype(int),
 'estimador__oob_score': [False,True],
 'estimador__random_state': np.linspace(10,100).astype(int),
}


# In[24]:


grid=RandomizedSearchCV(estimator=pipeline_Bagging,param_distributions=busqueda,scoring='roc_auc',cv=10,n_jobs=-1,random_state=42)
grid.fit(datos,datos[objetivo])


# In[25]:


grid.best_estimator_.steps


# In[35]:


pipeline_Bagging_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=5,
                     random_state=77, svd_solver='auto', tol=0.0,
                     whiten=False)),
    ('estimador',BaggingClassifier(base_estimator=None, bootstrap=True,
                                   bootstrap_features=False, max_features=1.0,
                                   max_samples=1.0, n_estimators=87, n_jobs=-1,
                                   oob_score=False, random_state=39, verbose=0,
                                   warm_start=False))
])


# In[27]:


pipeline_Knn.fit(datos,datos[objetivo])
pipeline_Knn.score(datos,datos[objetivo])


# In[28]:


pipeline_Knn.get_params()


# In[29]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(0,7).astype(int),
 'reductor_dim__random_state': np.linspace(1,100).astype(int),
  'estimador__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'estimador__leaf_size': np.linspace(30,130).astype(int),
 'estimador__n_neighbors':np.linspace(5,10).astype(int),
 'estimador__weights': ['uniform','distance']
}


# In[30]:


grid=RandomizedSearchCV(estimator=pipeline_Knn,param_distributions=busqueda,scoring='roc_auc',cv=10,random_state=42,n_jobs=-1)
grid.fit(datos,datos[objetivo])


# In[31]:


grid.best_estimator_.steps


# In[32]:


pipeline_Knn_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=6,
                     random_state=65, svd_solver='auto', tol=0.0,
                     whiten=False)),
    ('estimador',KNeighborsClassifier(algorithm='kd_tree', leaf_size=56,
                                      metric='minkowski', metric_params=None,
                                      n_jobs=-1, n_neighbors=8, p=2,
                                      weights='uniform'))
])


# In[37]:


resultados['estimador_AdaBoost_opti']=evalucion(pipeline_AdaBoost_opti,datos,datos[objetivo])
resultados['estimador_Bagging_opti']=evalucion(pipeline_Bagging_opti,datos,datos[objetivo])
resultados['estimador_Knn_opti']=evalucion(pipeline_Knn_opti,datos,datos[objetivo])


# In[38]:


ver_resultado()


# In[39]:


pipeline_Knn_opti.fit(datos,datos[objetivo])
pipeline_Knn_opti.score(datos,datos[objetivo])


# In[40]:


pipeline_AdaBoost_opti.fit(datos,datos[objetivo])
pipeline_AdaBoost_opti.score(datos,datos[objetivo])


# In[38]:


pipeline_AdaBoost_opti.score(datos,datos[objetivo])


# **Ensamblaje final**

# In[41]:


pipeline_Bagging_opti.fit(datos,datos[objetivo])
pipeline_Bagging_opti.score(datos,datos[objetivo])


# In[42]:


with open('Titanic.json','w') as fname:
    columnas_titanic=datos.columns.to_list()
    json5.dump(columnas_titanic,fname)
    fname.close()
dtypes_titanic=datos.dtypes
dtypes_titanic={col:datos[col].dtypes for col in datos.columns}
joblib.dump(dtypes_titanic,'Titanic_dtypes.pkl')
joblib.dump(pipeline_AdaBoost_opti,'Titanic_estimador.pkl')


# In[43]:


predicho_Baggin=pipeline_Bagging_opti.predict(datos)
predicho_AdaBoost=pipeline_AdaBoost_opti.fit(datos,datos[objetivo]).predict(datos)
predicho_Knn=pipeline_Knn_opti.fit(datos,datos[objetivo]).predict(datos)


# In[44]:


import matplotlib
matplotlib.rcParams['figure.figsize']=[12,12]
plt.plot(datos[objetivo],predicho_Baggin,marker='.',color='r',ls='--',lw=0.30)
plt.title('Relacion de datos real a predicho con Bagging')
plt.xlabel('Datos Real')
plt.ylabel('Valor predicho');


# In[45]:


plt.plot(datos[objetivo],predicho_AdaBoost,marker='.',color='g',ls='--',lw=0.30)
plt.title('Relacion de datos real a predicho con AdaBoost')
plt.xlabel('Datos Real')
plt.ylabel('Valor predicho');


# In[46]:


plt.plot(datos[objetivo],predicho_Knn,marker='.',color='b',ls='--',lw=0.30)
plt.title('Relacion de datos real a predicho con Knn')
plt.xlabel('Datos Real')
plt.ylabel('Valor predicho');


# In[7]:


rlf=joblib.load('Titanic_estimador.pkl')


# In[35]:


rlf.predict(datos)[:20]


# In[34]:


datos[objetivo][:20]


# In[ ]:




