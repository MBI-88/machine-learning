#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X_train=np.arange(10).reshape((10,1))
y_train=np.array([1.0,1.3,3.1,2.0,5.0,6.3,6.6,7.4,8.0,9.0])
plt.plot(X_train,y_train,'o',markersize=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[3]:


X_train_norm=(X_train-np.mean(X_train))/np.std(X_train)
ds_train_orig=tf.data.Dataset.from_tensor_slices((tf.cast(X_train_norm,tf.float32),tf.cast(y_train,tf.float32)))


# In[4]:


class Modelo(tf.keras.Model):
    def __init__(self):
        super(Modelo,self).__init__()
        self.W=tf.Variable(0.0,name='peso')
        self.b=tf.Variable(0.0,name='sesgo')
    
    def call(self,X):
        return self.W*X+self.b


# In[5]:


model=Modelo()
model.build(input_shape=(None,1))
model.summary()


# In[6]:


def lossis(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def model_train(modelo,X,y,rate):
    with tf.GradientTape() as tape:
        coste=lossis(y,modelo(X))
    dW,db=tape.gradient(coste,[model.W,model.b])
    model.W.assign_sub(rate*dW)
    model.b.assign_sub(rate*db)

num_epo=200
steps=100
batch_size=1
step_epo=int(np.ceil(len(y_train)/batch_size))

ds_train=ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size=batch_size)
Ws,bs=[],[]

for i,batch in enumerate(ds_train,1):
    if i >= step_epo*num_epo:
        break
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    bx,by=batch
    coste=lossis(by,model(bx))
    model_train(model,bx,by,rate=0.01)
    if i % steps==0:
        print('Epoca: {} Paso: {} Perdida: {:6.4f} '.format(i/step_epo,i,coste))


# In[7]:


print('Parametros finales {} {}'.format(model.W.numpy(),model.b.numpy()))


# In[8]:


X_test=np.linspace(0,9,num=100).reshape(-1,1)
X_test_nor=(X_test-np.mean(X_test))/np.std(X_test)
y_pred=model(tf.cast(X_test_nor,dtype=tf.float32))

fig=plt.figure(figsize=(13,5))
ax=fig.add_subplot(1,2,1)
plt.plot(X_train_norm,y_train,'o',markersize=10)
plt.plot(X_test_nor,y_pred,'--',lw=3)
plt.legend(['Training example','Linear Reg'],fontsize=15)
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
ax=fig.add_subplot(1,2,2)
plt.plot(Ws,lw=3)
plt.plot(bs,lw=3)
plt.legend(['Pesos','Sesgo'],fontsize=15)
ax.set_xlabel('Iteracion',size=15)
ax.set_ylabel('Value',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
plt.show()


# ## Usando los metodos de la API

# In[9]:


model=Modelo()
model.compile(optimizer='sgd',loss=lossis,metrics=['mae','mse'])


# In[10]:


model.fit(X_train_norm,y_train,epochs=num_epo,batch_size=batch_size,verbose=1)


# In[ ]:




