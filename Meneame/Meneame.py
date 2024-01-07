#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/meneame.csv').drop(['news_domain','submission_content'],axis=1)
datos.head()


# In[42]:


datos.shape


# In[43]:


datos.isnull().sum()


# In[144]:


variable_obj='votes'
col_num=datos.drop(variable_obj,axis=1).select_dtypes([np.number]).columns
col_cat='sub_name'


# In[179]:


from sklearn.model_selection import train_test_split,cross_validate,learning_curve,RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.externals import joblib
from sklearn.base import BaseEstimator,TransformerMixin
import json5
from scipy.sparse import issparse
from category_encoders import OneHotEncoder


# In[146]:


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
    
resultados={}
def valor_medio_absoluto(estimador,X,y):
    estimator=estimador.predict(X)
    return mean_absolute_error(y,estimator)
def evalucion(estimador,dato,objetivo):
    return cross_validate(estimador,dato,objetivo,scoring=valor_medio_absoluto,cv=15,n_jobs=-1,return_train_score=True)
def ver_resultado():
    dataframe=pd.DataFrame(resultados).T
    return print(dataframe)


# In[147]:


pipeline_num=Pipeline([
    ('selector',ColumnExtractor(columns=col_num)),
    ('imputador',SimpleImputer()),
    ('estandarizador',MaxAbsScaler())
])
pipeline_cat=Pipeline([
    ('selector',ColumnExtractor(columns=col_cat)),
    ('codificador',OneHotEncoder()),
    ('estandarizador',MaxAbsScaler())
])
pipeline_procesado=FeatureUnion([
    ('pipeline_num',pipeline_num),
    ('pipeline_cat',pipeline_cat),
])


# In[181]:


pipeline_Knn=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD()),
    ('estimador',KNeighborsRegressor(n_jobs=-1))
])
pipeline_Stacking=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD()),
    ('estimador',StackingCVRegressor(regressors=[DecisionTreeRegressor(),ExtraTreeRegressor()],random_state=42,cv=10,
                                     meta_regressor=XGBRegressor(n_jobs=-1)))
])
pipeline_DecisionTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD()),
    ('estimador',DecisionTreeRegressor())
])
pipeline_ExtraTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD()),
    ('estimador',ExtraTreeRegressor())
])
pipeline_Elastic=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador',ElasticNet())
])


# In[150]:


resultados['estimador_Knn']=evalucion(pipeline_Knn,datos,datos[variable_obj])
resultados['estimador_Stacking']=evalucion(pipeline_Stacking,datos,datos[variable_obj])


# In[151]:


ver_resultado()


# In[152]:


pipeline_Knn.fit(X_train,y_train)
pipeline_Knn.score(X_test,y_test)


# In[153]:


pipeline_Stacking.fit(X_train,y_train)
pipeline_Stacking.score(X_test,y_test)


# In[70]:


pipeline_Knn.get_params()


# In[154]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,8).astype(int),
 'reductor_dim__random_state': np.linspace(10,150).astype(int),
 'estimador__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'estimador__leaf_size':np.linspace(30,100).astype(int),
 'estimador__n_neighbors': np.linspace(5,15).astype(int),
 'estimador__p': [1,2],
 'estimador__weights': ['uniform','distance']
}


# In[155]:


random_search=RandomizedSearchCV(estimator=pipeline_Knn,param_distributions=busqueda,cv=10,random_state=42,n_iter=10,n_jobs=-1,scoring=valor_medio_absoluto)
random_search.fit(datos,datos[variable_obj])


# In[156]:


random_search.best_estimator_.steps


# In[157]:


pipeline_Knn_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=1, random_state=112,
      svd_solver='auto', tol=0.0, whiten=False)),
    ('estimador',KNeighborsRegressor(algorithm='ball_tree', leaf_size=38, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=12, p=2,
                      weights='distance'))
])


# In[171]:


resultados['estimador_Knn_opti']=evalucion(pipeline_Knn_opti,datos,datos[variable_obj])


# In[172]:


pipeline_Knn_opti.fit(datos,datos[variable_obj])
pipeline_Knn_opti.score(datos,datos[variable_obj])


# In[80]:


pipeline_Stacking.get_params()


# In[160]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,8).astype(int),
 'reductor_dim__random_state': np.linspace(10,150).astype(int),
 'estimador__meta_regressor__booster': ['gbtree','gblinear','dart'],
 #'estimador__meta_regressor__importance_type': ["gain","weight", "cover", "total_gain","total_cover"],
 #'estimador__meta_regressor__max_depth': np.linspace(3,6).astype(int),
 #'estimador__meta_regressor__n_estimators': np.linspace(100,150,10).astype(int),
 #'estimador__meta_regressor__random_state': np.linspace(10,100).astype(int),
 #'estimador__decisiontreeregressor__max_depth': np.linspace(3,6).astype(int),
 'estimador__decisiontreeregressor__random_state':np.linspace(10,100).astype(int) ,
 'estimador__extratreeregressor__max_depth': np.linspace(3,6).astype(int),
 'estimador__extratreeregressor__random_state': np.linspace(10,100).astype(int),
}


# In[168]:


random_search=RandomizedSearchCV(estimator=pipeline_Stacking,param_distributions=busqueda,scoring=valor_medio_absoluto,cv=10,n_iter=10,n_jobs=-1,random_state=42)
random_search.fit(datos,datos[variable_obj])
random_search.best_estimator_.steps


# In[169]:


pipeline_Stacking_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD(algorithm='randomized', n_components=1, n_iter=5, random_state=130,
               tol=0.0)),
    ('estimador',StackingCVRegressor(regressors=[ExtraTreeRegressor(ccp_alpha=0.0,
                                                     criterion='mse', max_depth=5,
                                                     max_features='auto',
                                                     max_leaf_nodes=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=2,
                                                     min_weight_fraction_leaf=0.0,
                                                     random_state=87,
                                                     splitter='random')],random_state=42,cv=10,
                                     meta_regressor=XGBRegressor(base_score=0.5,
                                                  booster='gblinear',
                                                  colsample_bylevel=1,
                                                  colsample_bynode=1,
                                                  colsample_bytree=1, gamma=0,
                                                  importance_type='gain',
                                                  learning_rate=0.1,
                                                  max_delta_step=0, max_depth=3,
                                                  min_child_weight=1,
                                                  missing=None, n_estimators=100,
                                                  n_jobs=-1, nthread=None,
                                                  objective='reg:linear',
                                                  random_state=0, reg_alpha=0,)))
])


# In[170]:


pipeline_Stacking_opti.fit(datos,datos[variable_obj])
pipeline_Stacking_opti.score(datos,datos[variable_obj])


# In[173]:


resultados['estimador_Stacking_opti']=evalucion(pipeline_Stacking_opti,datos,datos[variable_obj])


# In[174]:


ver_resultado()


# In[175]:


resultados['estimador_DecisionTree']=evalucion(pipeline_DecisionTree,datos,datos[variable_obj])
resultados['estimador_ExtraTree']=evalucion(pipeline_ExtraTree,datos,datos[variable_obj])


# In[176]:


ver_resultado()


# In[177]:


pipeline_DecisionTree.fit(datos,datos[variable_obj])
pipeline_DecisionTree.score(datos,datos[variable_obj])


# In[178]:


pipeline_ExtraTree.fit(datos,datos[variable_obj])
pipeline_ExtraTree.score(datos,datos[variable_obj])


# In[183]:


pipeline_Elastic.fit(datos,datos[variable_obj])
pipeline_Elastic.get_params()


# In[184]:


pipeline_Elastic.score(datos,datos[variable_obj])


# In[185]:


resultados['estimador_Elastic']=evalucion(pipeline_Elastic,datos,datos[variable_obj])
ver_resultado()


# In[188]:


busqueda={
 'estimador__alpha': np.linspace(0.001,1.0).astype(float),
 'estimador__copy_X': [True,False],
 'estimador__fit_intercept': [True,False],
 'estimador__l1_ratio': np.linspace(0.1,1.0).astype(float),
 'estimador__max_iter': np.linspace(1000,100000,10).astype(int),
 'estimador__normalize': [True,False],
 'estimador__positive': [True,False],
 'estimador__precompute': [True,False],
 'estimador__random_state': np.linspace(10,100).astype(int),
 'estimador__selection': ['cyclic','random'],
 'estimador__warm_start': [True,False]
}


# In[190]:


random_search=RandomizedSearchCV(pipeline_Elastic,param_distributions=busqueda,cv=10,random_state=42,n_iter=10,n_jobs=-1,scoring=valor_medio_absoluto)
random_search.fit(datos,datos[variable_obj])


# In[191]:


random_search.best_estimator_.steps


# In[192]:


pipeline_Elastic_opti=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('estimador',ElasticNet(alpha=0.9796122448979593, copy_X=False, fit_intercept=True,
             l1_ratio=0.21020408163265308, max_iter=23000, normalize=True,
             positive=True, precompute=False, random_state=63, selection='cyclic',
             tol=0.0001, warm_start=True))
])


# In[193]:


pipeline_Elastic_opti.fit(datos,datos[variable_obj])
pipeline_Elastic_opti.score(datos,datos[variable_obj])


# Ensamblado final

# In[194]:


with open('Meneame_columns.json','w') as fname:
    meneame_colums=datos.columns.to_list()
    json5.dump(meneame_colums,fname)
    fname.close()

meneame_dtypes=datos.dtypes
meneame_dtypes={col:datos[col].dtypes for col in datos.columns}
joblib.dump(meneame_dtypes,'Meneame_dtypes.pkl')
joblib.dump(pipeline_Knn_opti,'Meneame_estimador.pkl')


# In[201]:


datos[variable_obj][:20]


# In[202]:


pipeline_Knn_opti.predict(datos)[:20]


# In[ ]:




