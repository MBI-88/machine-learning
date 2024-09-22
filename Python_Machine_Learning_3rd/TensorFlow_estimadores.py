#!/usr/bin/env python
# coding: utf-8

# """ Pasos para la utilizacion de los estimadores de tensorflow
#     1- Definir una funcion de entrada para cargar los datos 
#     2- Convertir el data ser hacia  un a columna de variables
#     3- Instanciar el estimador de la biblioteca de tensorflow
#     4- Usar los metodos: train(),evaluate(),predict()
# """
# 

# In[1]:


import pandas as pd 
import tensorflow as tf 
import numpy as np 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


path='C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/Auto_MPG.csv'
data=pd.read_csv(path,sep=',',na_values='?')
data.head()


# In[3]:


data=data.dropna()
data=data.reset_index(drop=True)


# In[4]:


from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(data,train_size=0.80)
train_stats=df_train.describe().T
train_stats


# In[5]:


Colnum=[
    'Cylinders','Displacement','Horsepower','Weight','Acceleration'
]


# In[6]:


df_train_norm,df_test_norm=df_train.copy(),df_test.copy()

for col in Colnum:
    mean=train_stats.loc[col,'mean']
    std=train_stats.loc[col,'std']
    df_train_norm.loc[:,col]=(df_train_norm.loc[:,col]-mean)/std
    df_test_norm.loc[:,col]=(df_test_norm.loc[:,col]-mean)/std

df_train_norm.tail()


# In[7]:


num_feature=[]
for col in Colnum:
    num_feature.append(tf.feature_column.numeric_column(key=col))


# In[8]:


feature_year=tf.feature_column.numeric_column(key='ModelYear')
bucketized_feature=[]
bucketized_feature.append(tf.feature_column.bucketized_column(source_column=feature_year,boundaries=[73,76,79]))


# In[9]:


feature_origin=tf.feature_column.categorical_column_with_vocabulary_list(key='Origin',vocabulary_list=[1,2,3])

categorical_indicator_feature=[]
categorical_indicator_feature.append(tf.feature_column.indicator_column(feature_origin))# codificacion one hot a la variable origin


# In[10]:


def train_input_fn(df_train,batch_size=8):
    df=df_train.copy()
    train_x,train_y=df,df.pop('MPG')
    dataset=tf.data.Dataset.from_tensor_slices((dict(train_x),train_y))
    return dataset.shuffle(1000).repeat().batch(batch_size)


# In[11]:


ds=train_input_fn(df_train_norm)
batch=next(iter(ds))
print('Key: ',batch[0].keys())


# In[12]:


print('Batch Model: ',batch[0]['ModelYear'])


# In[13]:


def eval_input_fn(df_test,batch_size=8):
    df=df_test.copy()
    test_x,test_y=df,df.pop('MPG')
    dataset=tf.data.Dataset.from_tensor_slices((dict(test_x),test_y))
    return dataset.batch(batch_size)


# In[14]:


all_feature_columns=(
    num_feature+bucketized_feature+categorical_indicator_feature
)


# In[15]:


regresor=tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,hidden_units=[32,10],
)


# In[15]:


epoca=1000
batch_size=8
global_step=epoca*int(np.ceil(len(df_train)/batch_size))
print('Pasos globales: ',global_step)


# In[17]:


regresor.train(
    input_fn=lambda:train_input_fn(df_train_norm,batch_size=batch_size),
    steps=global_step)


# In[18]:


eval_=regresor.evaluate(
    input_fn=lambda: eval_input_fn(df_test_norm,batch_size=batch_size))

print('Average Loss: {:.4f}'.format(eval_['average_loss']))


# In[19]:


pred_rs=regresor.predict(
    input_fn=lambda: eval_input_fn(df_test_norm,batch_size=8))

print(next(iter(pred_rs)))


# In[20]:


boosted_tree=tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns,
    n_batches_per_layer=20,
    n_trees=200)

boosted_tree.train(
    input_fn=lambda:train_input_fn(df_train_norm,batch_size=batch_size))


# In[21]:


eval_=boosted_tree.evaluate(
    input_fn=lambda: eval_input_fn(df_test_norm,batch_size=8))

print('Average Loss:  {:.4f}'.format(eval_['average_loss']))


# ## Problema del XOR. Creando un estimador personalizado con keras model

# In[16]:


# Nota: Esta tecnica es utilizada para distribuir el trabajo entre equipos,  ademas nos brinda facilmente el guardado de los checkpoints del modelo para poder usarlos para un entrenamiento posterior

tf.random.set_seed(1)
np.random.seed(1)

# Creando el set de datos

x=np.random.uniform(low=-1,high=1,size=(200,2))
y=np.ones(len(x))
y[x[:,0]*x[:,1]<0]=0

x_train=x[:100,:]
y_train=y[:100]
x_val=x[100:,:]
y_val=y[100:]


# In[17]:


model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,),name='input_features'),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])


# In[18]:


def train_input_fn(x_train,y_train,batch_size=8):
    dataset=tf.data.Dataset.from_tensor_slices(({'input_features':x_train},y_train.reshape(-1,1)))
    return dataset.shuffle(100).repeat().batch(batch_size)

def eval_input_fn(x_val,y_val,batch_size=8):
    if y_val is None:
        dataset=tf.data.Dataset.from_tensor_slices(({'input_features':x_val}))
    else:
        dataset=tf.data.Dataset.from_tensor_slices(({'input_features':x_val},y_val.reshape(-1,1)))
    
    return dataset.batch(batch_size)

features=[
    tf.feature_column.numeric_column(key='input_features',shape=(2,))
]


# In[19]:


model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

my_estimator=tf.keras.estimator.model_to_estimator(
    keras_model=model)


# In[20]:


num_epochs=200
batch_size=2
step_epochs=np.ceil(len(x_train)/batch_size)

my_estimator.train(
    input_fn=lambda:train_input_fn(x_train,y_train,batch_size),
    steps=num_epochs*step_epochs
)


# In[21]:


my_estimator.evaluate(
    input_fn=lambda:eval_input_fn(x_val,y_val,batch_size))


# ## Usando el dataset mnist de sklearn

# In[22]:


from sklearn.datasets import load_digits


# In[23]:


dato=load_digits()
train_x=dato.data[:1000,:]
train_y=dato.target[:1000]
eval_x=dato.data[1000:,:]
eval_y=dato.target[1000:]

train_x.shape,train_y.shape,eval_x.shape,eval_y.shape


# In[24]:


feature_colum=[
    tf.feature_column.numeric_column(key='input_features',shape=(64,))
]


# In[25]:


regresor_clf=tf.estimator.DNNClassifier(
    feature_columns=feature_colum,
    n_classes=10,
    hidden_units=[24,16])


# In[26]:


num_epochs=1000
batch_size=20
step_epochs=int(np.ceil(len(train_y)/batch_size))

regresor_clf.train(
    input_fn=lambda:train_input_fn(train_x,train_y,batch_size),
    steps=num_epochs*step_epochs)


# In[27]:


result=regresor_clf.evaluate(
    input_fn=lambda:eval_input_fn(eval_x,eval_y,batch_size))

print(result)


# In[ ]:




