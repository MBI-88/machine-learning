#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/US_Accidents_May19.csv').drop(['ID','Source', 'TMC', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat','Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
       'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
       'Astronomical_Twilight', 'Airport_Code', 'Weather_Timestamp',
       'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
       'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump',
       'Crossing'],axis=1)
datos.columns


# In[ ]:


datos.sample(frac=0.00040,random_state=50).to_csv('US_accidents_May19_short.csv',index=False)


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

data=pd.read_csv('US_accidents_May19_short.csv')
data.head()


# In[2]:


data.shape


# In[3]:


#data['Amenity']=data['Amenity'].replace({False:0,True:1})
data['Side']=data['Side'].replace({'R':1,'L':0})
#data['Bump']=data['Bump'].replace({False:0,True:1})
#data['Crossing']=data['Crossing'].replace({False:0,True:1})
#datos['Give_Way']=datos['Give_Way'].replace({False:0,True:1})
#datos['Junction']=datos['Junction'].replace({False:0,True:1})
#datos['No_Exit']=datos['No_Exit'].replace({False:0,True:1})
#datos['Railway']=datos['Railway'].replace({False:0,True:1})
#datos['Roundabout']=datos['Roundabout'].replace({False:0,True:1})
#datos['Station']=datos['Station'].replace({False:0,True:1})
#data['Stop']=data['Stop'].replace({False:0,True:1})
#data['Traffic_Calming']=data['Traffic_Calming'].replace({False:0,True:1})
#data['Traffic_Signal']=data['Traffic_Signal'].replace({False:0,True:1})
#data['Turning_Loop']=data['Turning_Loop'].replace({False:0})
#data['Sunrise_Sunset']=data['Sunrise_Sunset'].replace({'Night':0,'Day':1})
#datos['Civil_Twilight']=datos['Civil_Twilight'].replace({'Night':0,'Day':1})
#datos['Nautical_Twilight']=datos['Nautical_Twilight'].replace({'Night':0,'Day':1})
#datos['Astronomical_Twilight']=datos['Astronomical_Twilight'].replace({'Night':0,'Day':1})


# In[4]:


data.head()


# In[5]:


variable_obj='Distance(mi)'
variable_num=data.drop(variable_obj,axis=1).select_dtypes([np.number]).columns
variable_text='Description'
variable_cat=data.drop(variable_text,axis=1).select_dtypes([np.object]).columns


# In[6]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate,RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA,TruncatedSVD
from category_encoders import OneHotEncoder
from scipy.sparse import issparse
import json5


# In[7]:


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


# In[41]:


resultados={}
def valor_medio_absoluto(estimador,X,y):
    estimator=estimador.predict(X)
    return mean_absolute_error(y,estimator)
def evalucion(estimador,dato,objetivo):
    return cross_validate(estimador,dato,objetivo,scoring=valor_medio_absoluto,cv=15,n_jobs=-1,return_train_score=True)
def ver_resultado():
    dataframe=pd.DataFrame(resultados).T
    return print(dataframe)


# In[15]:


pipeline_num=Pipeline([
    ('selector',ColumnExtractor(variable_num)),
    ('imputador',SimpleImputer()),
    ('estandarizador',MinMaxScaler())
])
pipeline_cat=Pipeline([
    ('selector',ColumnExtractor(variable_cat)),
    ('codificador',OneHotEncoder()),
    ('estandarizador',MinMaxScaler())
])
pipeline_tex=Pipeline([
    ('selector',ColumnExtractor(variable_text)),
    ('vectorizador',TfidfVectorizer()),
    ('transformador',DenseTransformer()),
    ('estandarizador',MinMaxScaler())
])
pipeline_procesado=FeatureUnion([
    ('pipeline_num',pipeline_num),
    ('pipeline_cat',pipeline_cat),
    ('pipeline_text',pipeline_tex)
])


# In[16]:


pipeline_ExtraTree=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('5_best_variables',SelectKBest(k=6)),
    ('estimador',ExtraTreesRegressor(n_jobs=-1))
])
pipeline_Gradiente=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('5_best_variables',SelectKBest(k=6)),
    ('estimador',GradientBoostingRegressor())
])
pipeline_Knn=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA()),
    ('5_best_variables',SelectKBest(k=6)),
    ('estimador',KNeighborsRegressor(n_jobs=-1))
])


# In[17]:


resultados['estimador_ExtraTree']=evalucion(pipeline_ExtraTree,data,data[variable_obj])
resultados['estimador_Gradiente']=evalucion(pipeline_Gradiente,data,data[variable_obj])
resultados['estimador_Knn']=evalucion(pipeline_Knn,data,data[variable_obj])


# In[18]:


ver_resultado()


# In[19]:


pipeline_Knn.fit(data,data[variable_obj])


# In[20]:


pipeline_Knn.score(data,data[variable_obj])


# In[21]:


pipeline_Knn.get_params()


# In[22]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,10).astype(int),
 'reductor_dim__random_state': np.linspace(10,100).astype(int),
 'estimador__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'estimador__leaf_size':np.linspace(30,100).astype(int),
 'estimador__n_neighbors': np.linspace(3,10).astype(int),
 'estimador__weights': ['uniform','distance']
}


# In[23]:


random_search=RandomizedSearchCV(pipeline_Knn,param_distributions=busqueda,random_state=42,cv=10,scoring=valor_medio_absoluto,n_iter=10,n_jobs=-1)
random_search.fit(data,data[variable_obj])


# In[24]:


random_search.best_estimator_.steps


# In[25]:


pipeline_Knn_opt=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',PCA(copy=True, iterated_power='auto', n_components=7, random_state=79,
      svd_solver='auto', tol=0.0, whiten=False)),
    ('5_best_variables',SelectKBest(k=6)),
    ('estimador',KNeighborsRegressor(algorithm='auto', leaf_size=90, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=4, p=2,
                      weights='uniform'))
])


# In[26]:


pipeline_Knn_opt.fit(data,data[variable_obj])
pipeline_Knn_opt.score(data,data[variable_obj])


# In[23]:


pipeline_ExtraTree.fit(data,data[variable_obj])
pipeline_ExtraTree.score(data,data[variable_obj])


# In[27]:


pipeline_Gradiente.fit(data,data[variable_obj])
pipeline_Gradiente.score(data,data[variable_obj])


# In[28]:


pipeline_Gradiente.get_params()


# In[29]:


busqueda={
 'reductor_dim':[PCA(),TruncatedSVD()],
 'reductor_dim__n_components': np.linspace(1,10).astype(int),
 'reductor_dim__random_state': np.linspace(10,100).astype(int),
 #'estimador__learning_rate':np.linspace(0.001,1.0).astype(float),
 #'estimador__loss': ['ls', 'lad', 'huber', 'quantile'],
 'estimador__max_depth': np.linspace(3,14).astype(int),
 #'estimador__max_features': np.linspace(1,5).astype(int),
 #'estimador__min_samples_split': np.linspace(2,5).astype(int),
 'estimador__n_estimators': np.linspace(10,150).astype(int),
 'estimador__random_state': np.linspace(10,100).astype(int),
}


# In[30]:


random_search=RandomizedSearchCV(pipeline_Gradiente,param_distributions=busqueda,random_state=42,cv=10,scoring=valor_medio_absoluto,n_iter=10,n_jobs=-1)
random_search.fit(data,data[variable_obj])


# In[32]:


random_search.best_estimator_.steps


# In[33]:


pipeline_Gradiente_opt=Pipeline([
    ('pipeline_procesado',pipeline_procesado),
    ('reductor_dim',TruncatedSVD(algorithm='randomized', n_components=6, n_iter=5, random_state=35,
               tol=0.0)),
    ('5_best_variables',SelectKBest(k=6)),
    ('estimador',GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                            init=None, learning_rate=0.1, loss='ls', max_depth=10,
                            max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=58,
                            n_iter_no_change=None, presort='deprecated',
                            random_state=59, subsample=1.0, tol=0.0001,
                            validation_fraction=0.1, verbose=0, warm_start=False))
])


# In[34]:


pipeline_Gradiente_opt.fit(data,data[variable_obj])
pipeline_Gradiente_opt.score(data,data[variable_obj])


# In[42]:


resultados['estimador_Knn_opt']=evalucion(pipeline_Knn_opt,data,data[variable_obj])
resultados['estimador_Gradiente_opt']=evalucion(pipeline_Gradiente_opt,data,data[variable_obj])


# In[43]:


ver_resultado()


# In[44]:


with open('US_Accidents_May19_columns.json','w') as fname:
    columnas_us=data.columns.to_list()
    json5.dump(columnas_us,fname)
    fname.close()
    
dtype_us=data.dtypes
dtype_us={col:data[col].dtypes for col in data.columns}
joblib.dump(dtype_us,'US_Accidents_May19_dtype.pkl')
joblib.dump(pipeline_Gradiente_opt,'US_Accidents_May19_estimator.pkl')


# In[45]:


dita={
    'fontsize': 18,
    'fontweight' : 14,
}
plt.plot(data[variable_obj],data['Start_Lng'],marker='.',color='b',ls='-',lw=0.30)
plt.title('Relacion de Distancia con Start_Lng',fontdict=dita,color='w')
plt.xlabel('Distancia',fontdict=dita,color='w')
plt.ylabel('Start_Lng',fontdict=dita,color='w');


# **Funcion para cambiar los datos de la columna Side**

# In[46]:


def Side(diccionario):
    side='Side'
    if side in diccionario:
        valor =diccionario[side]
        if valor =='R':
            diccionario[side]=1
            return diccionario
        else:
            diccionario[side]=0
            return diccionario
    else:
        pass
            


# In[47]:


prueba=pd.read_csv('US_accidents_May19_short.csv')
obs=prueba.to_dict(orient='record')[10]


# In[48]:


obs


# In[49]:


def dict_a_df(obs,columnas,dtypes):
    obs_t=Side(obs)
    obs_df=pd.DataFrame([obs_t])
    for col,dtype in dtypes.items():
        if col in obs_df.columns:
            obs_df[col]=obs_df[col].astype(dtype)
        else:
            obs_df[col]=None
    return obs_df


# In[50]:


with open('US_Accidents_May19_columns.json','r') as fname:
    columnas_us=json5.load(fname)
    fname.close()
    
dtype_us=joblib.load('US_Accidents_May19_dtype.pkl')
estimador=joblib.load('US_Accidents_May19_estimator.pkl')


# In[51]:


obs_dataframe=dict_a_df(obs,columnas_us,dtype_us)


# In[52]:


obs_dataframe


# In[53]:


estimador.predict(obs_dataframe)


# In[ ]:




