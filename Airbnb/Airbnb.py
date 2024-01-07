#!/usr/bin/env python
# coding: utf-8

# In[264]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/airbnb.csv')
datos.head()


# In[187]:


datos.shape


# In[188]:


datos['room_type'].unique()


# In[189]:


datos['room_type']=datos['room_type'].replace({'Entire home/apt':1,'Private room':2,'Shared room':3})


# In[190]:


datos.isnull().sum()


# In[191]:


from sklearn.model_selection import cross_val_score,train_test_split,RandomizedSearchCV,learning_curve
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.pipeline import Pipeline,FeatureUnion
import json5


# In[241]:


objetivo='price'
col_num=datos.drop(objetivo,axis=1).select_dtypes([np.number]).columns
col_cat=datos.drop(objetivo,axis=1).select_dtypes([np.object]).columns

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
    
resultados={}
def valor_medio_absoluto(estimador,X,y):
    estimator=estimador.predict(X)
    return mean_absolute_error(y,estimator)
def evalucion(estimador,dato,objetivo):
    return cross_val_score(estimador,dato,objetivo,scoring=valor_medio_absoluto,cv=10,n_jobs=-1).mean()


# In[242]:


pipeline_num=Pipeline([
    ('selector',ColumnExtractor(columns=col_num)),
    ('imputador',SimpleImputer(missing_values=np.nan,strategy='mean')),
    ('estandarizador',MinMaxScaler())
])
pipeline_cat=Pipeline([
    ('selector',ColumnExtractor(columns=col_cat)),
    ('codificador',OneHotEncoder(sparse=False)),
    ('estandarizador',MinMaxScaler())
])
pipeline_procesado=FeatureUnion([
    ('pipeline_num',pipeline_num),
    ('pipeline_cat',pipeline_cat)
])


# In[243]:


pipeline_Lasso=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador',Lasso())
])
pipeline_AdaBoost=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',AdaBoostRegressor())
])
pipeline_Bagging=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()), 
    ('estimador',BaggingRegressor(n_jobs=-1))
])
pipeline_RandomForest=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()), 
    ('estimador',RandomForestRegressor(n_jobs=-1))
])
pipeline_Knn=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('estimador',KNeighborsRegressor(n_jobs=-1))
])


# In[204]:


resultados['estimador_Lasso']=evalucion(pipeline_Lasso,datos,datos[objetivo])
resultados['estimador_AdaBoost']=evalucion(pipeline_AdaBoost,datos,datos[objetivo])
resultados['estimador_Bagging']=evalucion(pipeline_Bagging,datos,datos[objetivo])
resultados['estimador_RandomForest']=evalucion(pipeline_RandomForest,datos,datos[objetivo])
resultados['estimador_Knn']=evalucion(pipeline_Knn,datos,datos[objetivo])


# In[206]:


resultados


# In[213]:


pipeline_Lasso.fit(datos,datos[objetivo])
pipeline_Lasso.score(datos,datos[objetivo])


# In[214]:


pipeline_Bagging.fit(datos,datos[objetivo])
pipeline_Bagging.score(datos,datos[objetivo])


# In[215]:


pipeline_AdaBoost.fit(datos,datos[objetivo])
pipeline_AdaBoost.score(datos,datos[objetivo])


# In[216]:


pipeline_Knn.fit(datos,datos[objetivo])
pipeline_Knn.score(datos,datos[objetivo])


# In[217]:


pipeline_RandomForest.fit(X_train,y_train)
pipeline_RandomForest.score(X_test,y_test)


# In[77]:


pipeline_Knn.get_params()


# In[218]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,8).astype(int),
 'reductor_dim__random_state': np.linspace(10,150).astype(int),
 'estimador__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'estimador__leaf_size': np.linspace(30,130).astype(int),
 'estimador__n_neighbors': np.linspace(1,10).astype(int),
 'estimador__p': [1,2],
 'estimador__weights': ['uniform','distance']
}


# In[219]:


random_search=RandomizedSearchCV(pipeline_Knn,param_distributions=busqueda,scoring=valor_medio_absoluto,cv=10,n_iter=10,n_jobs=-1,random_state=42)
random_search.fit(datos,datos[objetivo])


# In[220]:


random_search.best_estimator_.steps


# In[244]:


pipeline_Knn_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=1, random_state=38,
      svd_solver='auto', tol=0.0, whiten=False)),
    ('estimador',KNeighborsRegressor(algorithm='brute', leaf_size=113, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=1, p=1,
                      weights='uniform'))
])


# In[245]:


pipeline_Knn_opti.fit(datos,datos[objetivo])
pipeline_Knn_opti.score(datos,datos[objetivo])


# In[223]:


resultados['estimador_Knn_pti']=evalucion(pipeline_Knn_opti,X_train,y_train)


# In[86]:


pipeline_RandomForest.get_params()


# In[224]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,8).astype(int),
 'reductor_dim__random_state': np.linspace(10,150).astype(int),
 'estimador__bootstrap': [True,False],
 'estimador__max_depth': np.linspace(3,10).astype(int),
 'estimador__n_estimators': np.linspace(100,1000).astype(int),
 'estimador__random_state': np.linspace(10,150).astype(int),
}


# In[225]:


random_search=RandomizedSearchCV(pipeline_RandomForest,param_distributions=busqueda,scoring=valor_medio_absoluto,cv=10,n_iter=10,n_jobs=-1,random_state=42)
random_search.fit(datos,datos[objetivo])


# In[226]:


random_search.best_estimator_.steps


# In[227]:


pipeline_RandomForest_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD(algorithm='randomized', n_components=4, n_iter=5, random_state=115,
               tol=0.0)), 
    ('estimador',RandomForestRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                        max_depth=9, max_features='auto', max_leaf_nodes=None,
                        max_samples=None, min_impurity_decrease=0.0,
                        min_impurity_split=None, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        n_estimators=191, n_jobs=-1, oob_score=False,
                        random_state=15, verbose=0, warm_start=False))
])


# In[228]:


pipeline_RandomForest_opti.fit(X_train,y_train)
pipeline_RandomForest_opti.score(X_test,y_test)


# In[229]:


resultados['estimador_RandomForest_opti']=evalucion(pipeline_RandomForest_opti,X_train,y_train)


# In[232]:


resultados


# In[233]:


pipeline_Bagging.get_params()


# In[234]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,8).astype(int),
 'reductor_dim__random_state': np.linspace(10,150).astype(int),
 'estimador__bootstrap': [True,False],
 'estimador__bootstrap_features': [True,False],
 'estimador__n_estimators': np.linspace(10,300,10).astype(int),
 'estimador__oob_score': [True,False],
 'estimador__random_state': np.linspace(10,150).astype(int),
 'estimador__warm_start': [True,False]
}


# In[235]:


random_search=RandomizedSearchCV(pipeline_Bagging,param_distributions=busqueda,cv=10,n_iter=10,scoring=valor_medio_absoluto,random_state=42,n_jobs=-1)
random_search.fit(datos,datos[objetivo])


# In[236]:


random_search.best_estimator_.steps


# In[237]:


pipeline_Bagging_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=1, random_state=130,
      svd_solver='auto', tol=0.0, whiten=False)), 
    ('estimador',BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False,
                   max_features=1.0, max_samples=1.0, n_estimators=106, n_jobs=-1,
                   oob_score=True, random_state=98, verbose=0, warm_start=False))
])


# In[238]:


pipeline_Bagging_opti.fit(datos,datos[objetivo])
pipeline_Bagging_opti.score(datos,datos[objetivo])


# In[239]:


resultados['estimador_Bagging_opti']=evalucion(pipeline_Bagging_opti,datos,datos[objetivo])
resultados


# In[246]:


with open('Airbnb_columns.json','w') as fname:
    columns_air=datos.columns.to_list()
    json5.dump(columns_air,fname)
    fname.close()

dtypes_air=datos.dtypes
dtypes_air={col:datos[col].dtypes for col in datos.columns}
joblib.dump(dtypes_air,'Airbnb_dtypes.pkl')
joblib.dump(pipeline_Knn_opti,'Airbnb_estimator.pkl')


# In[258]:


def room_type(obs):
    columna='room_type'
    vl_1,vl_2,vl_3='Entire home/apt','Private room','Shared room'
    if columna in obs:
        valor=obs[columna]
        if valor == vl_1:
            obs[columna]=1
            return obs
        if valor == vl_2:
            obs[columna]=2
            return obs
        if valor == vl_3:
            obs[columna]=3
            return obs
    else:
      return obs

def dict_to_df(diccionario,columnas,dtypes):
    df=room_type(diccionario)
    df=pd.DataFrame([diccionario])
    for col,dtype in dtypes.items():
        if col in df.columns:
            df[col]=df[col].astype(dtype)
        else:
            df[col]=None
    return df


# In[250]:


with open('Airbnb_columns.json','r') as fname:
    columns_air=json5.load(fname)
    fname.close()

dtypes_air=joblib.load('Airbnb_dtypes.pkl')
rlf=joblib.load('Airbnb_estimator.pkl')


# In[285]:


obs=datos.to_dict(orient='record')[10]
obs


# In[286]:


obs_df=dict_a_df(obs,columns_air,dtypes_air)
obs_df


# In[287]:


rlf.predict(obs_df)


# In[288]:


pipeline_Bagging_opti.predict(obs_df)


# In[ ]:




