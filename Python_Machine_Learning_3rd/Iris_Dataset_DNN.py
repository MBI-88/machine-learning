#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.datasets import load_iris
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


data=load_iris(return_X_y=False,as_frame=False)
data.data[:10]


# In[16]:


X_train,y_train,X_test,y_test=train_test_split(data['data'],data['target'],train_size=0.80)
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# # DNN

# In[17]:


iris_model=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation='sigmoid',name='fc1',input_shape=(4,)),
    tf.keras.layers.Dense(3,name='fc2',activation='softmax')
])
iris_model.summary()


# In[18]:


iris_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[19]:


num_epo=100
batch=2
step_epo=int(np.ceil(len(y_train)/batch))
historial=iris_model.fit(X_train,X_test,epochs=num_epo,steps_per_epoch=step_epo,verbose=1,batch_size=2)


# In[20]:


hist=historial.history

fig=plt.figure(figsize=(12,5))
ax=fig.add_subplot(1,2,1)
ax.plot(hist['loss'],lw=3)
ax.set_title('Perdidas',size=15)
ax.set_xlabel('Epoca',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
ax=fig.add_subplot(1,2,2)
ax.plot(hist['accuracy'],lw=3)
ax.set_title('Accuracy',size=15)
ax.set_xlabel('Epoca',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
plt.show()


# In[21]:


result=iris_model.evaluate(y_train,y_test,batch_size=2)
print('Perdida: {:.4f}  Accuracy: {:.4f} '.format(*result))


# In[22]:


iris_model.save('iris_class.h5',overwrite=True,include_optimizer=True,save_format='h5')


# In[23]:


carga_modelo=tf.keras.models.load_model('iris_class.h5')
carga_modelo.summary()


# In[24]:


result=carga_modelo.evaluate(y_train,y_test,batch_size=2,verbose=0)
print('Perdida: {:.4f}  Accuracy: {:.4f} '.format(*result))


# In[ ]:




