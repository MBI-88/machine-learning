#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from mlxtend.plotting import plot_decision_regions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Modalidad # 1

# In[2]:


modelo=tf.keras.Sequential()
modelo.add(
    tf.keras.layers.Dense(units=16,activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.glorot_uniform(),
    bias_initializer=tf.keras.initializers.Constant(2.0)))
modelo.add(
    tf.keras.layers.Dense(units=32,activation=tf.keras.activations.sigmoid,
    kernel_regularizer=tf.keras.regularizers.l1)
)
modelo.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()
              ]
)



# ## Modalidad # 2

# In[3]:


# Resolviendo el problema de clasificacion XOR

tf.random.set_seed(1)
np.random.seed(1)
X=np.random.uniform(low=-1,high=1,size=(200,2))
y=np.ones(len(X))
y[X[:,0]*X[:,1]<0]=0


# In[4]:


X_train=X[:100,:]
y_train=y[:100]
X_eval=X[100:,:]
y_eval=y[100:]

fig=plt.figure(figsize=(8,8))
plt.plot(X[y==0,0],X[y==0,1],'o',alpha=0.75,markersize=10)
plt.plot(X[y==1,0],X[y==1,1],'<',alpha=0.75,markersize=10)
plt.xlabel(r'$x_1$',size=15)
plt.ylabel(r'$x_2$',size=15)
plt.show()


# In[5]:


# Crando el modelo 

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,input_shape=(2,),activation='sigmoid'))

model.summary()


# In[6]:


model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

hist=model.fit(X_train,y_train,validation_data=(X_eval,y_eval),epochs=200,batch_size=2,verbose=0)


# In[7]:


historial=hist.history

fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(historial['loss'],lw=4)
plt.plot(historial['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(historial['binary_accuracy'],lw=4)
plt.plot(historial['val_binary_accuracy'],lw=4)
plt.legend(['Train Acc','Validation Acc'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_eval,y=y_eval.astype(np.integer),clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


# In[8]:


new_model=tf.keras.Sequential()
new_model.add(tf.keras.layers.Dense(units=8,input_shape=(2,),activation='relu'))
new_model.add(tf.keras.layers.Dense(units=8,activation='relu'))
#new_model.add(tf.keras.layers.Dense(units=4,activation='relu'))
new_model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

new_model.summary()


# In[9]:


new_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()])

hist=new_model.fit(X_train,y_train,validation_data=(X_eval,y_eval),epochs=200,batch_size=2,verbose=0)


# In[10]:


historial=hist.history

fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(historial['loss'],lw=4)
plt.plot(historial['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(historial['binary_accuracy'],lw=4)
plt.plot(historial['val_binary_accuracy'],lw=4)
plt.legend(['Train Acc','Validation Acc'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_eval,y=y_eval.astype(np.integer),clf=new_model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


# In[11]:


np.array(historial['binary_accuracy']).mean()


# In[12]:


np.array(historial['val_binary_accuracy']).mean()


# ## Modalidad # 3

# In[13]:


inputs=tf.keras.Input(shape=(2,))
# Hidden layers
h1=tf.keras.layers.Dense(units=4,activation='relu')(inputs)
h2=tf.keras.layers.Dense(units=4,activation='relu')(h1)
h3=tf.keras.layers.Dense(units=4,activation='relu')(h2)
# Ouput layers
output=tf.keras.layers.Dense(units=1,activation='sigmoid')(h3)

m1=tf.keras.Model(inputs=inputs,outputs=output)
m1.summary()


# In[14]:


m1.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

hist=m1.fit(
    X_train,y_train,validation_data=(X_eval,y_eval),epochs=200,batch_size=2,verbose=0
)


# In[15]:


historial=hist.history

fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(historial['loss'],lw=4)
plt.plot(historial['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(historial['binary_accuracy'],lw=4)
plt.plot(historial['val_binary_accuracy'],lw=4)
plt.legend(['Train Acc','Validation Acc'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_eval,y=y_eval.astype(np.integer),clf=m1)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


# In[16]:


np.array(historial['binary_accuracy']).mean()


# In[17]:


np.array(historial['val_binary_accuracy']).mean()


# ## Modalidad # 4

# In[20]:


class Modelo(tf.keras.Model):
    def __init__(self):
        super(Modelo,self).__init__()
        self.h_1=tf.keras.layers.Dense(units=4,activation='relu')
        self.h_2=tf.keras.layers.Dense(units=4,activation='relu')
        self.h_3=tf.keras.layers.Dense(units=4,activation='relu')
        self.outputs_layer=tf.keras.layers.Dense(units=1,activation='sigmoid')

    def call(self,inputs):# Es importate el uso de call cuando se usa esta modalidad
        h=self.h_1(inputs)
        h=self.h_2(h)
        h=self.h_3(h)
        return self.outputs_layer(h)
    
capa=Modelo()
capa.build(input_shape=(None,2))
capa.summary()


# In[21]:


capa.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()])
    
hist=capa.fit(X_train,y_train,validation_data=(X_eval,y_eval),epochs=200,batch_size=2,verbose=0)


# In[22]:


historial=hist.history

fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(historial['loss'],lw=4)
plt.plot(historial['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(historial['binary_accuracy'],lw=4)
plt.plot(historial['val_binary_accuracy'],lw=4)
plt.legend(['Train Acc','Validation Acc'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_eval,y=y_eval.astype(np.integer),clf=capa)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


# ## Creando capa personalizada 

# In[30]:


class Noislayer(tf.keras.layers.Layer):
    def __init__(self,output_dim,noise_stddev=0.1,**kwargs):
        self.output_dim=output_dim
        self.noise_stddev=noise_stddev
        super(Noislayer,self).__init__(**kwargs)

    def build(self,input_shape):
        self.w=self.add_weight(name='weigths',shape=(input_shape[1],self.output_dim),initializer='random_normal',trainable=True)
        self.b=self.add_weight(shape=(self.output_dim,),initializer='zeros',trainable=True)

    def call(self,inputs,training=False):
        if training:
            batch=tf.shape(inputs)[0]
            dim=tf.shape(inputs)[1]
            noise=tf.random.normal(shape=(batch,dim),mean=0.0,stddev=self.noise_stddev)
            noisy_inputs=tf.add(inputs,noise)
        else:
            noisy_inputs=inputs
        z=tf.matmul(noisy_inputs,self.w)+self.b 
        return tf.keras.activations.relu(z)

    def get_config(self):
        config=super(Noislayer,self).get_config()
        config.update({'output_dim':self.output_dim,'noise_stddev':self.noise_stddev})
        return config


# In[31]:


# Probando la capa personalizada

capa_per=Noislayer(4)
capa_per.build(input_shape=(None,4))
x=tf.zeros(shape=(1,4))
tf.print(capa_per(x,training=True))


# In[33]:


config=capa_per.get_config()
new_capa=Noislayer.from_config(config)
tf.print(new_capa(x,training=True))


# In[34]:


nn=tf.keras.Sequential([
    Noislayer(4,noise_stddev=0.1),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid'),])

nn.build(input_shape=(None,2))


# In[35]:


nn.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

hist=nn.fit(X_train,y_train,validation_data=(X_eval,y_eval),epochs=200,batch_size=2,verbose=0)


# In[36]:


historial=hist.history

fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(historial['loss'],lw=4)
plt.plot(historial['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(historial['binary_accuracy'],lw=4)
plt.plot(historial['val_binary_accuracy'],lw=4)
plt.legend(['Train Acc','Validation Acc'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_eval,y=y_eval.astype(np.integer),clf=nn)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


# In[37]:


print(np.array(historial['binary_accuracy']).mean())


# In[38]:


print(np.array(historial['val_binary_accuracy']).mean())


# In[ ]:




