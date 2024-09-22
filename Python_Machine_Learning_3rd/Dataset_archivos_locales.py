#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib 
import matplotlib.pyplot as plt  
import tensorflow as tf 
get_ipython().run_line_magic('matplotlib', 'inline')

image=pathlib.Path('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images')
file_list=sorted([str(item) for item in image.glob('*.jpg')])
print(file_list)


# In[2]:


fig=plt.figure(figsize=(10,5))
for i, file in enumerate(file_list):
    img_raw=tf.io.read_file(file)
    img=tf.image.decode_image(img_raw)
    #img=tf.image.resize(img,[120,80])
    #img /=255.0
    print('Image sahpe: ',img.shape)
    ax=fig.add_subplot(2,3,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file),size=10)
plt.tight_layout()
plt.show()


# In[3]:


labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)


# In[4]:


ds_file=tf.data.Dataset.from_tensor_slices((file_list,labels))
for item in ds_file:
    print(item[0].numpy(),item[1].numpy())


# In[5]:


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image, label

img_width, img_height = 64, 64
ds_file_image=ds_file.map(load_and_preprocess)
fig=plt.figure(figsize=(10,5))
for i , exa in enumerate(ds_file_image):
    print(exa[0].shape, exa[1].numpy())
    ax=fig.add_subplot(2,3,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(exa[0])
    ax.set_title('{}'.format(exa[1].numpy()),size=10)
plt.tight_layout()
plt.show()


# In[14]:


for i,y in ds_file_image.take(5):
    print(i.shape,' ',y)


# In[7]:


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(2,2),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Conv2D(64,(2,2),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2))
])


# In[8]:


model.compute_output_shape(input_shape=(None,64,64,3))


# In[9]:


model.add(tf.keras.layers.Flatten())
model.compute_output_shape(input_shape=(None,64,64,3))


# In[10]:


model.add(tf.keras.layers.Dense(units=16384,activation='relu'))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
model.build(input_shape=(None,64,64,3))
model.summary()


# In[11]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)


# In[12]:


ds_file_image=ds_file_image.shuffle(buffer_size=50).batch(batch_size=3).repeat(count=None)
hist=model.fit(ds_file_image,epochs=5,steps_per_epoch=20)


# # Nota : Solo para probar

# In[ ]:




