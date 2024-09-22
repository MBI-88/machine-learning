#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import  time
import itertools
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


x_train=x_train[:784,:]
x_train.shape


# In[4]:


x_train=tf.data.Dataset.from_tensor_slices(x_train)
def prepro(ex,mode='uniform'):
    image=ex
    image=tf.image.convert_image_dtype(image,tf.float32)
    image=tf.reshape(image,[-1])
    image=image*2-1.0
    if mode=='uniform':
        input_z=tf.random.uniform(shape=(20,),minval=-1.0,maxval=1.0)
    elif mode=='normal':
        input_z=tf.random.normal(shape=(20,))
    return input_z,image

train_set=x_train
train_set=train_set.map(prepro)


# In[5]:


train_set=train_set.batch(32,drop_remainder=True)
input_z,input_real=next(iter(train_set))
print('Input_z: ',input_z.shape)
print('Input_real: ',input_real.shape)


# In[6]:


# Creando el modelo

# Funcion para el generador 
def make_generator_network():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=100,use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(units=784,activation='tanh'))
    return model

def make_discriminator_network():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=100))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=1,activation=None))
    return model


# In[7]:


gen_model=make_generator_network()
gen_model.build(input_shape=(None,20))
gen_model.summary()


# In[8]:


disc_model=make_discriminator_network()
disc_model.build(input_shape=(None,784))
disc_model.summary()


# In[9]:


g_output=gen_model(input_z)
print(g_output.shape)


# In[10]:


d_logits_real=disc_model(input_real)
d_logits_fake=disc_model(g_output)
print('Real: ',d_logits_real.shape)
print('Fake: ',d_logits_fake.shape)


# In[11]:


loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)

g_labels_real=tf.ones_like(d_logits_fake)
g_loss=loss_fn(y_true=g_labels_real,y_pred=d_logits_fake)
print('Generator Loss: {:.4f}'.format(g_loss))


# In[12]:


d_labels_real=tf.ones_like(d_logits_real)
d_labels_fake=tf.zeros_like(d_logits_fake)

d_loss_real=loss_fn(y_true=d_labels_real,y_pred=d_logits_real)
d_loss_fake=loss_fn(y_true=d_labels_fake,y_pred=d_logits_fake)

print('Discriminator Losses: Real {:.4f} Fake {:.4f}'.format(d_loss_real.numpy(),d_loss_fake.numpy()))


# In[18]:


num_epochs=300
batch_size=64
image_size=(28,28)
tf.random.set_seed(1)
np.random.seed(1)
fixed_z=tf.random.uniform(shape=(64,20),minval=-1.0,maxval=1.0)

def create_sample(g_model,input_z):
    g_output=g_model(input_z,training=False)
    images=tf.reshape(g_output,(batch_size,*image_size))
    return (images+1)/2.0

# Estableciendo el dataset

mnist_set=x_train
mnist_set=mnist_set.map(lambda ex: prepro(ex))


# In[19]:


mnist_set=mnist_set.shuffle(10000)
mnist_set=mnist_set.batch(batch_size,drop_remainder=True)


# In[20]:


# Estableciendo el modelo
with tf.device('/device:gpu:0'):
    gen_model=make_generator_network()
    gen_model.build(input_shape=(None,20))

    disc_model=make_discriminator_network()
    disc_model.build(input_shape=(None,np.prod(image_size)))

loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer=tf.keras.optimizers.Adam()
d_optimizer=tf.keras.optimizers.Adam()
all_losses=[]
all_d_vals=[]
epoch_sample=[]
start_time=time.time()
for epoch in range(1,num_epochs+1):
    epoch_losses,epoch_d_vals=[],[]
    for i,(input_z,input_real) in enumerate(mnist_set):
        with tf.GradientTape() as g_tape:
            g_output=gen_model(input_z)
            d_logits_fake=disc_model(g_output,training=True)
            labels_real=tf.ones_like(d_logits_fake)
            g_loss=loss_fn(y_true=labels_real,y_pred=d_logits_fake)
        
        g_grads=g_tape.gradient(g_loss,gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads,gen_model.trainable_variables))

        with tf.GradientTape() as d_tape:
            d_logits_real=disc_model(input_real,training=True)
            d_labels_real=tf.ones_like(d_logits_real)
            d_loss_real=loss_fn(y_true=d_labels_real,y_pred=d_logits_real)
            d_logits_fake=disc_model(g_output,training=True)
            d_labels_fake=tf.zeros_like(d_logits_fake)
            d_loss_fake=loss_fn(y_true=d_labels_fake,y_pred=d_logits_fake)
            d_loss=d_loss_real+d_loss_fake
        
        d_grads=d_tape.gradient(d_loss,disc_model.trainable_variables)
        
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads,disc_model.trainable_variables))

        epoch_losses.append(
            (g_loss.numpy(),d_loss.numpy(),d_loss_real.numpy(),d_loss_fake.numpy()))
        
        d_probs_real=tf.reduce_mean(tf.sigmoid(d_loss_real))
        d_probs_fake=tf.reduce_mean(tf.sigmoid(d_loss_fake))
        epoch_d_vals.append((d_probs_real.numpy(),d_probs_fake.numpy()))
        all_losses.append(epoch_losses)
        all_d_vals.append(epoch_d_vals)
        print(
            'Epoch {:03d} - ET {:.2f} min - Avg Losses >> ''G/D {:.4f}/{:.4f} [D-Real: {:.4f}  D-Fake {:.4f}]'.format(
             epoch,(time.time()-start_time)/60,*list(np.mean(all_losses[-1],axis=0))   ))
        epoch_sample.append(create_sample(gen_model,fixed_z).numpy())


# In[26]:


fig=plt.figure(figsize=(16,6))
ax=fig.add_subplot(1,2,1)
g_losses=[item[0] for item in itertools.chain(*all_losses)]
d_losses=[item[1]/2.0 for item in itertools.chain(*all_losses)]
plt.plot(g_losses,label='Generation loss',alpha=0.95)
plt.plot(d_losses,label='Discriminator loss',alpha=0.95)
plt.legend(fontsize=20)

ax.set_xlabel('Iteration',size=15)
ax.set_ylabel('Loss',size=15)

epoch=np.arange(1,101)
epoch2iter=lambda e: e*len(all_losses[-1])
epoch_ticks=[1,20,40,60,80,100,150,250,300]
newpos=[epoch2iter(e) for e in epoch_ticks]
ax2=ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward',60))
ax2.set_xlabel('Epoch',size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both',which='major',labelsize=15)
ax2.tick_params(axis='both',which='major',labelsize=15)

ax=fig.add_subplot(1,2,2)
d_vals_real=[item[0] for item in itertools.chain(*all_d_vals)]
d_vals_fake=[item[1] for item in itertools.chain(*all_d_vals)]
plt.plot(d_vals_real,alpha=0.75,label=r'Real: $D(G(\mathbf{x}))$')
plt.plot(d_vals_fake,alpha=0.75,label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration',size=15)
ax.set_ylabel('Discriminator ouput',size=15)

ax2=ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward',60))
ax2.set_xlabel('Epoch',size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both',which='major',labelsize=15)
ax2.tick_params(axis='both',which='major',labelsize=15)
plt.show()


# In[25]:


selected_epochs=[1,2,4,10,50,100,150,250,300]
fig=plt.figure(figsize=(10,14))
for i,e in enumerate(selected_epochs):
    for j in range(5):
        ax=fig.add_subplot(9,5,i*5+j+1)
        ax.set_xticks([])
        ax.set_yticks([])

        if j==0:
            ax.text(
                -0.06,0.5,'Epoch {} '.format(e),rotation=90,size=18,color='red',
                horizontalalignment='right',verticalalignment='center',transform=ax.transAxes)
        image=epoch_sample[e-1][j]
        ax.imshow(image,cmap='gray_r')

plt.show()


# # DCGAN

# In[1]:


import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import time
import itertools
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


# Modelo 
def make_dcgan_generator(z_size=20,output_size=(28,28,1),n_filters=128,n_blocks=2):
    size_factor=2**n_blocks
    hidden_size=(output_size[0]//size_factor,output_size[1]//size_factor)

    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_size,)),
        tf.keras.layers.Dense(units=n_filters*np.prod(hidden_size),use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((hidden_size[0],hidden_size[1],n_filters)),
        tf.keras.layers.Conv2DTranspose(filters=n_filters,kernel_size=(5,5),strides=(1,1),padding='same',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])

    nf=n_filters
    for i in range(n_blocks):
        nf=nf//2
        model.add(
            tf.keras.layers.Conv2DTranspose(filters=nf,kernel_size=(5,5),strides=(2,2),padding='same',use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
    
    model.add(
        tf.keras.layers.Conv2DTranspose(filters=output_size[2],kernel_size=(5,5),strides=(1,1),padding='same',use_bias=False,activation='tanh'))
    
    return model


def make_dcgan_discriminator(input_size=(28,28,1),n_filters=64,n_blocks=2):

    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.layers.Conv2D(filters=n_filters,kernel_size=5,strides=(1,1),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()])
    nf=n_filters
    for i in range(n_blocks):
        nf=nf*2
        model.add(
            tf.keras.layers.Conv2D(filters=nf,kernel_size=(5,5),strides=(2,2),padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=(7,7),padding='valid'))

    return model


# In[3]:


(train_image,train_label),(_,_)=tf.keras.datasets.mnist.load_data()
train_image=train_image.reshape(train_image.shape[0],28,28,1)
train_image=train_image[:784,:]
train_image.shape


# In[4]:


mnist_set=tf.data.Dataset.from_tensor_slices(train_image)


# In[5]:


z_size=20
def new_prepro(ex,mode='uniform'):
    image=ex
    image=tf.image.convert_image_dtype(image,tf.float32)
    image=image*2-1.0
    if mode=='uniform':
        input_z=tf.random.uniform(shape=(z_size,),minval=-1.0,maxval=1.0)
    elif mode=='normal':
        input_z=tf.random.normal(shape=(z_size,))
    return input_z,image


# In[6]:


num_epochs=50
batch_size=256
image_size=(28,28)
tf.random.set_seed(1)
np.random.seed(1)
fixed_z=tf.random.uniform(shape=(256,20),minval=-1.0,maxval=1.0)

def create_sample(g_model,input_z):
    g_output=g_model(input_z,training=False)
    images=tf.reshape(g_output,(batch_size,*image_size))
    return (images+1)/2.0

mnist_set=mnist_set.map(lambda ex:new_prepro(ex))


# In[7]:


mnist_set=mnist_set.shuffle(60000)
mnist_set=mnist_set.batch(batch_size,drop_remainder=True)


# In[8]:


gen_model=make_dcgan_generator()
gen_model.summary()


# In[9]:


disc_model=make_dcgan_discriminator()
disc_model.summary()


# In[10]:


loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer=tf.keras.optimizers.Adam()
d_optimizer=tf.keras.optimizers.Adam()
all_losses=[]
all_d_vals=[]
epoch_sample=[]
start_time=time.time()
for epoch in range(1,num_epochs+1):
    epoch_losses,epoch_d_vals=[],[]
    for i,(input_z,input_real) in enumerate(mnist_set):
        with tf.GradientTape() as g_tape:
            g_output=gen_model(input_z)
            d_logits_fake=disc_model(g_output,training=True)
            labels_real=tf.ones_like(d_logits_fake)
            g_loss=loss_fn(y_true=labels_real,y_pred=d_logits_fake)
        
        g_grads=g_tape.gradient(g_loss,gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads,gen_model.trainable_variables))

        with tf.GradientTape() as d_tape:
            d_logits_real=disc_model(input_real,training=True)
            d_labels_real=tf.ones_like(d_logits_real)
            d_loss_real=loss_fn(y_true=d_labels_real,y_pred=d_logits_real)
            d_logits_fake=disc_model(g_output,training=True)
            d_labels_fake=tf.zeros_like(d_logits_fake)
            d_loss_fake=loss_fn(y_true=d_labels_fake,y_pred=d_logits_fake)
            d_loss=d_loss_real+d_loss_fake
        
        d_grads=d_tape.gradient(d_loss,disc_model.trainable_variables)
        
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads,disc_model.trainable_variables))

        epoch_losses.append(
            (g_loss.numpy(),d_loss.numpy(),d_loss_real.numpy(),d_loss_fake.numpy()))
        
        d_probs_real=tf.reduce_mean(tf.sigmoid(d_loss_real))
        d_probs_fake=tf.reduce_mean(tf.sigmoid(d_loss_fake))
        epoch_d_vals.append((d_probs_real.numpy(),d_probs_fake.numpy()))
        all_losses.append(epoch_losses)
        all_d_vals.append(epoch_d_vals)
        print(
            'Epoch {:03d} - ET {:.2f} min - Avg Losses >> ''G/D {:.4f}/{:.4f} [D-Real: {:.4f}  D-Fake {:.4f}]'.format(
             epoch,(time.time()-start_time)/60,*list(np.mean(all_losses[-1],axis=0))   ))
        epoch_sample.append(create_sample(gen_model,fixed_z).numpy())


# In[11]:


fig=plt.figure(figsize=(16,6))
ax=fig.add_subplot(1,2,1)
g_losses=[item[0] for item in itertools.chain(*all_losses)]
d_losses=[item[1]/2.0 for item in itertools.chain(*all_losses)]
plt.plot(g_losses,label='Generation loss',alpha=0.95)
plt.plot(d_losses,label='Discriminator loss',alpha=0.95)
plt.legend(fontsize=20)

ax.set_xlabel('Iteration',size=15)
ax.set_ylabel('Loss',size=15)

epoch=np.arange(1,101)
epoch2iter=lambda e: e*len(all_losses[-1])
epoch_ticks=[1,20,40,60,80,100,150,250,300]
newpos=[epoch2iter(e) for e in epoch_ticks]
ax2=ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward',60))
ax2.set_xlabel('Epoch',size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both',which='major',labelsize=15)
ax2.tick_params(axis='both',which='major',labelsize=15)

ax=fig.add_subplot(1,2,2)
d_vals_real=[item[0] for item in itertools.chain(*all_d_vals)]
d_vals_fake=[item[1] for item in itertools.chain(*all_d_vals)]
plt.plot(d_vals_real,alpha=0.75,label=r'Real: $D(G(\mathbf{x}))$')
plt.plot(d_vals_fake,alpha=0.75,label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration',size=15)
ax.set_ylabel('Discriminator ouput',size=15)

ax2=ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward',60))
ax2.set_xlabel('Epoch',size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both',which='major',labelsize=15)
ax2.tick_params(axis='both',which='major',labelsize=15)
plt.show()


# In[17]:


selected_epochs=[1,2,4,10,50]
fig=plt.figure(figsize=(10,14))
for i,e in enumerate(selected_epochs):
    for j in range(5):
        ax=fig.add_subplot(5,5,i*5+j+1)
        ax.set_xticks([])
        ax.set_yticks([])

        if j==0:
            ax.text(
                -0.06,0.5,'Epoch {} '.format(e),rotation=90,size=18,color='red',
                horizontalalignment='right',verticalalignment='center',transform=ax.transAxes)
        image=epoch_sample[e-1][j]
        ax.imshow(image,cmap='gray_r')

plt.show()


# # WGAN-GP

# In[20]:


import tensorflow as tf 
import time 
import matplotlib.pyplot as plt 
import numpy as np 
import itertools 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[21]:


# Modelo 
def make_dcgan_generator(z_size=20,output_size=(28,28,1),n_filters=128,n_blocks=2):
    size_factor=2**n_blocks
    hidden_size=(output_size[0]//size_factor,output_size[1]//size_factor)

    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_size,)),
        tf.keras.layers.Dense(units=n_filters*np.prod(hidden_size),use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((hidden_size[0],hidden_size[1],n_filters)),
        tf.keras.layers.Conv2DTranspose(filters=n_filters,kernel_size=(5,5),strides=(1,1),padding='same',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])

    nf=n_filters
    for i in range(n_blocks):
        nf=nf//2
        model.add(
            tf.keras.layers.Conv2DTranspose(filters=nf,kernel_size=(5,5),strides=(2,2),padding='same',use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
    
    model.add(
        tf.keras.layers.Conv2DTranspose(filters=output_size[2],kernel_size=(5,5),strides=(1,1),padding='same',use_bias=False,activation='tanh'))
    
    return model


def make_dcgan_discriminator(input_size=(28,28,1),n_filters=64,n_blocks=2):

    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.layers.Conv2D(filters=n_filters,kernel_size=5,strides=(1,1),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()])
    nf=n_filters
    for i in range(n_blocks):
        nf=nf*2
        model.add(
            tf.keras.layers.Conv2D(filters=nf,kernel_size=(5,5),strides=(2,2),padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=1,kernel_size=(7,7),padding='valid'))

    return model


# In[22]:


(train_image,train_label),(_,_)=tf.keras.datasets.mnist.load_data()
train_image=train_image.reshape(train_image.shape[0],28,28,1)
train_image=train_image[:20000,:]
train_image.shape


# In[23]:


mnist_set=tf.data.Dataset.from_tensor_slices(train_image)


# In[24]:


def new_prepro(ex,mode='uniform'):
    image=ex
    image=tf.image.convert_image_dtype(image,tf.float32)
    image=image*2-1.0
    if mode=='uniform':
        input_z=tf.random.uniform(shape=(z_size,),minval=-1.0,maxval=1.0)
    elif mode=='normal':
        input_z=tf.random.normal(shape=(z_size,))
    return input_z,image

def create_sample(g_model,input_z):
    g_output=g_model(input_z,training=False)
    images=tf.reshape(g_output,(batch_size,*image_size))
    return (images+1)/2.0


# In[25]:


gen_model=make_dcgan_generator()
gen_model.summary()


# In[26]:


disc_model=make_dcgan_discriminator()
disc_model.summary()


# In[27]:


num_epochs=100
batch_size=128
image_size=(28,28)
z_size=20
lambda_gp=10.0
tf.random.set_seed(1)
np.random.seed(1)
mnist_set=mnist_set.map(lambda ex:new_prepro(ex))
mnist_set=mnist_set.shuffle(20000)
mnist_set=mnist_set.batch(batch_size,drop_remainder=True)

g_optimizer=tf.keras.optimizers.Adam(0.0002)
d_optimizer=tf.keras.optimizers.Adam(0.0002)
all_losses=[]
all_d_vals=[]
epoch_sample=[]
fixed_z=tf.random.uniform(shape=(batch_size,z_size),minval=-1.0,maxval=1.0)

@tf.function
def Operacion(input_z,input_real):
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        g_output=gen_model(input_z,training=True)
        d_critical_real=disc_model(input_real,training=True)
        d_critical_fake=disc_model(g_output,training=True)

        g_loss=-tf.math.reduce_mean(d_critical_fake)

        d_loss_real=-tf.math.reduce_mean(d_critical_real)
        d_loss_fake=tf.math.reduce_mean(d_critical_fake)
        d_loss=d_loss_real + d_loss_fake

        with tf.GradientTape() as gp_tape:
            alpha=tf.random.uniform(shape=[d_critical_real.shape[0],1,1,1],minval=0.0,maxval=1.0)
            interpolated=(alpha*input_real+(1-alpha)*g_output)
            gp_tape.watch(interpolated)
            d_critical_intp=disc_model(interpolated)
        
        grads_inp=gp_tape.gradient(d_critical_intp,[interpolated,])[0]
        grads_inp_l2=tf.sqrt(
            tf.reduce_sum(tf.square(grads_inp),axis=[1,2,3]))
        grads_penalty=tf.reduce_mean(tf.square(grads_inp_l2-1.0))

        d_loss=d_loss +lambda_gp*grads_penalty
    

    d_grads=d_tape.gradient(d_loss,disc_model.trainable_variables)
    d_optimizer.apply_gradients(grads_and_vars=zip(d_grads,disc_model.trainable_variables))

    g_grads=g_tape.gradient(g_loss,gen_model.trainable_variables)
    g_optimizer.apply_gradients(grads_and_vars=zip(g_grads,gen_model.trainable_variables))

    return (g_loss,d_loss,d_loss_real,d_loss_fake)
    


# In[28]:


start_time=time.time()
for epoch in range(1,num_epochs+1):
    epoch_losses=[]

    for i ,(input_z,input_real) in enumerate(mnist_set):
        g_loss,d_loss,d_loss_real,d_loss_fake=Operacion(input_z,input_real)
        epoch_losses.append((g_loss.numpy(),d_loss.numpy(),d_loss_real.numpy(),d_loss_fake.numpy()))
    
    all_losses.append(epoch_losses)
    print(
            'Epoch {:03d} - ET {:.2f} min - Avg Losses >> ''G/D {:6.2f}/{:.4f} [D-Real: {:6.2f}  D-Fake {:.4f}]'.format(
             epoch,(time.time()-start_time)/60,*list(np.mean(all_losses[-1],axis=0))   ))
    epoch_sample.append(create_sample(gen_model,fixed_z).numpy())


# In[29]:


selected_epochs=[1,2,4,10,50,100]
fig=plt.figure(figsize=(10,14))
for i,e in enumerate(selected_epochs):
    for j in range(5):
        ax=fig.add_subplot(6,5,i*5+j+1)
        ax.set_xticks([])
        ax.set_yticks([])

        if j==0:
            ax.text(
                -0.06,0.5,'Epoch {} '.format(e),rotation=90,size=18,color='red',
                horizontalalignment='right',verticalalignment='center',transform=ax.transAxes)
        image=epoch_sample[e-1][j]
        ax.imshow(image,cmap='gray_r')

plt.show()


# In[ ]:




