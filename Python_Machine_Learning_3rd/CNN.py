#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np 

def convolution1D(x,w,p=0,s=1):
    w_rot=np.array(w[::-1])
    x_padded=np.array(x)
    if p > 0:
        zero_pad=np.zeros(shape=p)
        x_padded=np.concatenate([zero_pad,x_padded,zero_pad])
    res=[]

    for i in range(0,int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)


# In[13]:


x=[1,3,2,4,5,6,1,3]
w=[1,0,3,1,2]
print('Convolucion ',convolution1D(x,w,p=2,s=1))


# In[19]:


print('Numpy resultado same ',np.convolve(x,w,mode='same'))
print('Numpy resultado full ',np.convolve(x,w,mode='full'))
print('Numpy resultado valid ',np.convolve(x,w,mode='valid'))


# ## Convolucion en 2 Dimensiones

# In[10]:


import scipy.signal as sc 

def convolution2D(X,W,p=(0,0),s=(1,1)):
    W_rot=np.array(W) [::-1,::-1]
    X_orig=np.array(X)
    n1=X_orig.shape[0]+2*p[0]
    n2=X_orig.shape[1]+2*p[1]
    X_padding=np.zeros(shape=(n1,n2))
    X_padding[p[0]:p[0]+X_orig.shape[0],p[1]:p[1]+X_orig.shape[1]]=X_orig
    res=[]
    for i in range(0,int((X_padding.shape[0]-W_rot.shape[0])/s[0])+1,s[0]):
        res.append([])
        for j in range(0,int((X_padding.shape[1]-W_rot.shape[1])/s[1])+1,s[1]):
            X_sub=X_padding[i:i+W_rot.shape[0],j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub*W_rot))
    return (np.array(res))

X=[[1,3,2,4],[5,6,1,3],[1,2,0,2],[3,4,3,2]]
W=[[1,0,3],[1,2,1],[0,1,1]]

print('Convolution_2D \n',convolution2D(X,W,p=(1,1),s=(1,1)))


# In[11]:


print('Convolution_2D Scipy \n',sc.convolve2d(X,W,mode='same'))


# ## Implementacion CNN

# In[1]:


from sklearn.datasets import load_digits
import tensorflow as tf 
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dato=load_digits()
train_x=dato.images[:800,:].reshape((800,8,8,1))
train_y=dato.target[:800]
eval_x=dato.images[800:1200,:].reshape((400,8,8,1))
eval_y=dato.target[800:1200]
test_x=dato.images[1200:,:].reshape((597,8,8,1))
test_y=dato.target[1200:]
train_x.shape,train_y.shape,eval_x.shape,eval_y.shape,test_x.shape,test_y.shape


# In[20]:


def my_input_fn(X,y,batch_size=32,shuffle=True,num_epochs=None):
    x_tensor=tf.cast(X,tf.float32)/255.0
    y_tensor=tf.cast(y,tf.int32)
    ds=tf.data.Dataset.from_tensor_slices((x_tensor,y_tensor))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(y))
    ds=ds.batch(batch_size=batch_size).repeat(num_epochs)
    return ds

set_train=my_input_fn(train_x,train_y,batch_size=12)
set_val= my_input_fn(eval_x,eval_y,shuffle=False,batch_size=12)
set_test= my_input_fn(test_x,test_y,shuffle=False,batch_size=12)


# In[21]:


for i,y in set_test.take(5):
    print(i.shape,'   ',y)


# In[14]:


mnist=tf.keras.Sequential()
mnist.add(tf.keras.layers.Convolution2D(
    filters=32,kernel_size=(5,5),strides=1,padding='same',data_format='channels_last',
    name='conv_1',activation='relu'))

mnist.add(tf.keras.layers.MaxPool2D(
    pool_size=2,name='pool_1'))

mnist.add(tf.keras.layers.Convolution2D(
    filters=64,kernel_size=(5,5),strides=(1,1),
    padding='same',name='conv_2',activation='relu'))

mnist.add(tf.keras.layers.MaxPool2D(
    pool_size=(2,2),name='pool_2'))


# In[15]:


mnist.compute_output_shape(input_shape=(None,8,8,1))


# In[16]:


mnist.add(tf.keras.layers.Flatten())
mnist.compute_output_shape(input_shape=(None,8,8,1))


# In[17]:


mnist.add(tf.keras.layers.Dense(
    units=256,name='fc_1',activation='relu'))

mnist.add(tf.keras.layers.Dropout(rate=0.5))

mnist.add(tf.keras.layers.Dense(
    units=10,name='fc_2',activation='softmax'))


# In[18]:


tf.random.set_seed(1)
mnist.build(input_shape=(None,8,8,1))
mnist.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


# In[19]:


epo=10
steps_epo=int(np.ceil(len(train_y)/12))
hist=mnist.fit(set_train,epochs=epo,
validation_data=set_val,shuffle=True,steps_per_epoch=epo*steps_epo,validation_steps=epo*steps_epo)


# In[20]:


historia=hist.history
x_arr=np.arange(len(historia['loss']))+1

fig=plt.figure(figsize=(10,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,historia['loss'],'-o',label='Train_loss')
ax.plot(x_arr,historia['val_loss'],'--<',label='Validation_loss')
ax.legend(fontsize=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,historia['accuracy'],'-o',label='Train_acc')
ax.plot(x_arr,historia['val_accuracy'],'--<',label='Validation_acc')
ax.legend(fontsize=15)
plt.show()


# In[21]:


result=mnist.evaluate(set_test,steps=epo*steps_epo)
print(result)


# In[22]:


batch_test=next(iter((set_test)))
preds=mnist(batch_test[0])
tf.print(preds.shape)


# In[23]:


preds=tf.argmax(preds,axis=1)
print(preds)


# In[24]:


fig=plt.figure(figsize=(12,4))
for i in range(12):
    ax=fig.add_subplot(2,6,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    img=batch_test[0][i,:,:,0]
    ax.imshow(img,cmap='gray_r')
    ax.text(0.9,0.1,'{}'.format(preds[i]),size=15,color='blue',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)

plt.show()


# ## CNN (Celebridades)

# In[1]:


import pandas as pd 
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=fetch_lfw_people(min_faces_per_person=20,resize=0.7)


# In[3]:


data.target_names[data.target[:16]]
def dibujar_caras(array_caras):
    fig,axes=plt.subplots(2,8,figsize=(25,10),subplot_kw={'xticks':(),'yticks':()})
    for target , image, ax in zip(data.target[:16],array_caras,axes.ravel()):
        ax.imshow(image,cmap='gray')
        ax.set_title(data.target_names[target])
    

dibujar_caras(data.images[:16])


# In[4]:


data.images.shape,data.target.shape


# In[5]:


data.target_names.shape,data.target_names


# In[6]:


X_train=data.images[:350,:].reshape((350,87,65,1))
y_train=data.target[:350]
X_eval=data.images[350:500,:].reshape((150,87,65,1))
y_eval=data.target[350:500]
X_test=data.images[500:,:].reshape((31,87,65,1))
y_test=data.target[500:]
X_train.shape,X_eval.shape,X_test.shape


# In[7]:


# Funcion para la argumentacion de datos en imagenes

def preproces(X,y,size=(28,28),mode='train'):
    if mode=='train':
        image_cropped=tf.image.random_crop(X,size=(X.shape[0],50,28,1))
        image_resized=tf.image.resize(image_cropped,size=size)
        image_flip=tf.image.random_flip_left_right(image_resized)
        return image_flip/255.0,tf.cast(y,tf.int32)
    else:
        image_cropped=tf.image.crop_to_bounding_box(X,offset_height=20,offset_width=0,
        target_height=28,target_width=28)
        image_resized=tf.image.resize(image_cropped,size=size)
        return image_resized/255.0,tf.cast(y,tf.int32)


# In[8]:


def my_input_Image(X,y,batch_size=32,shuffle=True,num_epoch=None,modo='train'):
    x,t=preproces(X,y,mode=modo)
    ds=tf.data.Dataset.from_tensor_slices((x,t))
    if shuffle:
        ds=ds.shuffle(buffer_size=1000)
    ds=ds.batch(batch_size=batch_size).repeat(num_epoch)
    return ds

training=my_input_Image(X_train,y_train,batch_size=32)
evaluate=my_input_Image(X_eval,y_eval,shuffle=False,modo='x',batch_size=32)
test=my_input_Image(X_test,y_test,shuffle=False,modo='x',batch_size=32)


# In[9]:


for i , y in test.take(5):
    print(i.shape,' ',y)


# In[10]:


## Creando la red
 
lrw=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.15),

    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2))])


# In[11]:


lrw.compute_output_shape(input_shape=(None,28,28,1))


# In[12]:


lrw.add(tf.keras.layers.Flatten())
lrw.compute_output_shape(input_shape=(None,28,28,1))


# In[13]:


lrw.add(tf.keras.layers.Dense(units=1152,activation='relu'))
lrw.add(tf.keras.layers.Dense(units=5,activation='softmax'))
tf.random.set_seed(1)
lrw.build(input_shape=(None,28,28,1))
lrw.summary()


# In[14]:


lrw.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


# In[15]:


epo=20
batch_size=32
steps_epo=int(np.ceil(len(X_train)/batch_size))
history=lrw.fit(training,epochs=epo,steps_per_epoch=epo*steps_epo,validation_data=evaluate,validation_steps=epo*steps_epo)


# In[15]:


historia=history.history
x_arr=np.arange(len(historia['loss']))+1

fig=plt.figure(figsize=(10,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,historia['loss'],'-o',label='Train_loss')
ax.plot(x_arr,historia['val_loss'],'--<',label='Validation_loss')
ax.legend(fontsize=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,historia['accuracy'],'-o',label='Train_acc')
ax.plot(x_arr,historia['val_accuracy'],'--<',label='Validation_acc')
ax.legend(fontsize=15)
plt.show()


# # No hay suficientes datos de imagenes para el modelo

# In[ ]:




