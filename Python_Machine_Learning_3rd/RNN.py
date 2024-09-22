#!/usr/bin/env python
# coding: utf-8

# # Implementacion del proceso de recurrencia (de una RNN)

# In[1]:


import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)


# In[2]:


rnn_layer=tf.keras.layers.SimpleRNN(
    units=2,use_bias=True,return_sequences=True)

rnn_layer.build(input_shape=(None,None,5))
w_xh,w_oo,b_h=rnn_layer.weights

print('W_xh shape :',w_xh.shape)
print('W_oo shape :',w_oo.shape)
print('b_h shape  :',b_h.shape)


# In[3]:


x_seq=tf.convert_to_tensor(
    [[1.0]*5,[2.0]*5,[3.0]*5],dtype=tf.float32)

output=rnn_layer(tf.reshape(x_seq,shape=(1,3,5)))
out_man=[]
for  t in range(len(x_seq)):
    xt=tf.reshape(x_seq[t],(1,5))
    print('Time step {} =>'.format(t))
    print(' Input    :',xt.numpy())
    ht=tf.matmul(xt,w_xh)+b_h
    print('  Hidden  :',ht.numpy())

    if t>0:
        prev_o=out_man[t-1]
    else:
        prev_o=tf.zeros(shape=(ht.shape))
    ot=ht+tf.matmul(prev_o,w_oo)
    ot=tf.math.tanh(ot)
    out_man.append(ot)
    print('  Output (manual)  :',ot.numpy())
    print('  SimpleRNN output   :'.format(t),output[0][t].numpy())
    print()


# # Procesado de los datos

# In[2]:


import tarfile  # Para desempaquetar datos comprimidos
import pandas as pd 
import os 
import numpy as np 

path='C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/aclImdb_v1.tar.gz'

#with tarfile.open(path,'r:gz') as tar:
    #tar.extractall()
  


# In[3]:


"""basepath='C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/aclImdb'
label={'pos':1,'neg':0}

data=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path_dir=os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path_dir)):
            with open(os.path.join(path_dir,file),'r',encoding='utf-8') as infile:
                txt=infile.read()
            data=data.append([[txt,label[l]]],ignore_index=True)

data.columns=['review','sentiment']  """


# In[4]:


"""np.random.seed(0)
data=data.reindex(np.random.permutation(data.index))
data.to_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movie_data_review.csv',index=False,encoding='utf-8')"""


# In[5]:


dataset=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movie_data_review.csv',encoding='utf-8')
dataset.tail(5)


# In[6]:


dataset.shape


# # Pasos para llevar el dataset al formato de una Rnn:
# # 1-Crear un tensorflow dataset y separar en entrenamiento, prueba y validacion
# # 2-Identificar las palabras unicas en el dataset
# # 3-Mapiar cada palabra unica hacia un entero unico y codificar el texto en un entero 
# # 4-Dividir el dataset en pequeños bloques para  intoroducirlo en el modelo 

# In[7]:


# Preparando el dataset
import tensorflow_datasets as tfd 
from collections import Counter

# Paso 1:
target=dataset.pop('sentiment')
ds_raw=tf.data.Dataset.from_tensor_slices((dataset.values,target.values))

# Verificando:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[:20],ex[1])


# In[8]:


ds_raw=ds_raw.shuffle(50000,reshuffle_each_iteration=False)
ds_raw_test=ds_raw.take(25000)
ds_raw_train_valid=ds_raw.skip(25000)
ds_raw_train=ds_raw_train_valid.take(20000)
ds_raw_valid=ds_raw_train_valid.skip(20000)


# Paso 2:
tokenizer=tfd.features.text.Tokenizer()# Nota: features.text.Tokenizer() -> deprecated.text.Tokenizer()
token_counter=Counter()
for example in ds_raw_train:
    tokens=tokenizer.tokenize(example[0].numpy()[0])
    token_counter.update(tokens)
print('Vocab_size :',len(token_counter))


# In[9]:


# Paso 3:
# Recortando la secuencia de datos de entrada para  usar una simple RNN
 
encoder=tfd.features.text.TokenTextEncoder(token_counter)

def encode(text_tensor,label):
    text=text_tensor.numpy()[0]
    encoded_text=encoder.encode(text)  
    return encoded_text,label
    
def encode_map_fn(text,label):
    return tf.py_function(encode,inp=[text,label],Tout=(tf.int64,tf.int64))



ds_train=ds_raw_train.map(encode_map_fn)
ds_valid=ds_raw_valid.map(encode_map_fn)
ds_test=ds_raw_test.map(encode_map_fn)

# Forma de algunos ejemplos
for example in ds_train.shuffle(1000).take(5):
    print('Longitud de secuencia :',example[0].shape)


# In[10]:


## Tomando una pequeña  muestra 
ds_subset=ds_train.take(8)
for example in ds_subset:
    print('Tamaño individual :',example[0].shape)


# In[11]:


# Dividiendo el dataset en bloques 
ds_batched=ds_subset.padded_batch(4,padded_shapes=([-1],[]))
for batch in ds_batched:
    print('Batch dimension :',batch[0].shape)


# In[12]:


train_data=ds_train.padded_batch(32,padded_shapes=([-1],[]))
valid_data=ds_valid.padded_batch(32,padded_shapes=([-1],[]))
test_data=ds_test.padded_batch(32,padded_shapes=([-1],[]))


# In[18]:


# Ejemplo de cracion de una RNN 
embedding_dim=20
vacab_size=len(token_counter) + 2
modelo=tf.keras.Sequential()
modelo.add(tf.keras.layers.Embedding(input_dim=vacab_size,output_dim=embedding_dim,name='embed_layer'))
modelo.add(tf.keras.layers.SimpleRNN(embedding_dim,return_sequences=True))
modelo.add(tf.keras.layers.SimpleRNN(embedding_dim))
modelo.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
modelo.summary()


# In[24]:


modelo.compile(
     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)
history=modelo.fit(train_data,validation_data=valid_data,epochs=10,steps_per_epoch=50,validation_steps=40)


# In[21]:


embedding_dim=20
vacab_size=len(token_counter) + 2

bi_lstm_model=tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vacab_size,output_dim=embedding_dim,name='embed_layer'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,name='lstm_layer',return_sequences=True),name='bidir_lstm'),

    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

bi_lstm_model.summary()


# In[22]:


bi_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[23]:


history=bi_lstm_model.fit(train_data,validation_data=valid_data,epochs=10) # El error esta en la cantidad de elementos de palabras unicas que se le pasa por la  capacidad  de la GPU


# In[24]:


result=bi_lstm_model.evaluate(test_data)
print('Test Acc : {:.2f}%'.format(result[1]*100))


# In[13]:


# Recortando la secuencia de datos de entrada para  usar una simple RNN. Resolviendo el problema de la capacidad de la GPU, disminuyendo el tamaño  muestral

def prepocesss_data(ds_raw_train,ds_raw_valid,ds_raw_test,max_seq_length=None,batch_size=32):
    tokenizer=tfd.features.text.Tokenizer()
    token_counst=Counter()
    for example in ds_raw_train:
        tokens=tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None :
            tokens=tokens[-max_seq_length:]
        token_counst.update(tokens)
    
    encoder=tfd.features.text.TokenTextEncoder(token_counst)
    def encode_max(text_tensor,label):
        text=text_tensor.numpy()[0]
        encoded_text=encoder.encode(text)
        if max_seq_length is not None:
            encoded_text=encoded_text[-max_seq_length:]
        return encoded_text,label
    
    def encoder_map_fn(text,label):
        return tf.py_function(encode_max,inp=[text,label],Tout=(tf.int64,tf.int64))

    ds_train=ds_raw_train.map(encoder_map_fn)
    ds_valid=ds_raw_valid.map(encoder_map_fn)
    ds_test=ds_raw_test.map(encoder_map_fn)

    train_data=ds_train.padded_batch(32,padded_shapes=([-1],[]))
    valid_data=ds_valid.padded_batch(32,padded_shapes=([-1],[]))
    test_data=ds_test.padded_batch(32,padded_shapes=([-1],[]))
    return (train_data,valid_data,test_data,len(token_counst))


# In[14]:


def build_rnn_model(embedding_dim,vocab_size,recurrent_type='SimpleRNN',n_recurrent_unit=64,n_recurrent_layers=1,bidirectional=True):
    model=tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,name='embed-layer'))
    for i in range(n_recurrent_layers):
        return_sequences=(i < n_recurrent_layers-1)
        if recurrent_type=='SimpleRNN':
            recurrent_layer=tf.keras.layers.SimpleRNN(
                units=n_recurrent_unit,return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))

        elif  recurrent_type=='LSTM':
            recurrent_layer=tf.keras.layers.LSTM(
                units=n_recurrent_unit,return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))

        elif recurrent_type=='GRU':
            recurrent_layer=tf.keras.layers.GRU(
                units=n_recurrent_unit,return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))

        if bidirectional:
            recurrent_layer=tf.keras.layers.Bidirectional(
                recurrent_layer,name='birdir-'+ recurrent_layer.name)
        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    return model


# In[15]:


batch_size=32
embedding_dim=20
max_seq_length=200
train_data,valid_data,test_data,n=prepocesss_data(ds_raw_train,ds_raw_valid,ds_raw_test,max_seq_length=max_seq_length,batch_size=batch_size)

vocab_size=n+2


# In[18]:


rnn_model=build_rnn_model(
    embedding_dim,vocab_size,recurrent_type='SimpleRNN',
    n_recurrent_unit=64,n_recurrent_layers=1,bidirectional=True
)
rnn_model.summary()


# In[21]:


rnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=rnn_model.fit(train_data,validation_data=valid_data,epochs=10,steps_per_epoch=50,validation_steps=40)


# In[22]:


result=rnn_model.evaluate(test_data)
print('Test Acc : {:.2f}%'.format(result[1]*100))


# In[16]:


rnn_model=build_rnn_model(
    embedding_dim,vocab_size,recurrent_type='LSTM',
    n_recurrent_unit=64,n_recurrent_layers=1,bidirectional=True
)
rnn_model.summary()


# In[17]:


rnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=rnn_model.fit(train_data,validation_data=valid_data,epochs=10)


# In[18]:


result=rnn_model.evaluate(test_data)
print('Test Acc : {:.2f}%'.format(result[1]*100))


# In[19]:


rnn_model=build_rnn_model(
    embedding_dim,vocab_size,recurrent_type='GRU',
    n_recurrent_unit=64,n_recurrent_layers=1,bidirectional=True
)
rnn_model.summary()
rnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=rnn_model.fit(train_data,validation_data=valid_data,epochs=10)


# In[20]:


result=rnn_model.evaluate(test_data)
print('Test Acc : {:.2f}%'.format(result[1]*100)) # El mejor funcionamiento en prediccion


# # Modelado del Lenguaje

# In[44]:


import numpy as np 
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[45]:


# Preparando los datos

with open('1268-0.txt','r',encoding='utf-8') as fp: # Nota se tuvo que  codificar  en el formato utf-8 para decodificar las caracteres
    text=fp.read()
start=text.find('THE MYSTERIOUS ISLAND')
end_index=text.find('End of the Project Gutenberg')
text=text[start:end_index]
char_set=set(text)
print('Total Length  :',len(text))
print('\n')
print('Unique Characters  :',len(char_set))


# # Nota: se necesita crear un mapear caracteres a enteros para  pasalos al modelo y mapaear en revarsa para obtener la ligadura de la prediccion conlos caracteres del  mapa

# In[46]:


char_sorted=sorted(char_set)
char2int={ch:i for i,ch in enumerate(char_sorted)}
char_array=np.array(char_sorted)
text_encoded=np.array(
    [char2int[ch] for ch in text],dtype=np.int32
)
print('Text  encoded shape :',text_encoded.shape)


# In[47]:


print(text[:15],'== Encoding ==>',text_encoded[:15])
print(text_encoded[15:21],'== Reverse ==>',''.join(char_array[text_encoded[15:21]]))


# In[48]:


ds_text_encode=tf.data.Dataset.from_tensor_slices(text_encoded)
for ex in ds_text_encode.take(5):
    print('{} -> {}'.format(ex.numpy(),char_array[ex.numpy()]))


# In[49]:


seq_length=40
chunk_size=seq_length+1
ds_chunk=ds_text_encode.batch(chunk_size,drop_remainder=True)

def split_input_target(chunk):
    input_seq=chunk[:-1]
    target_seq=chunk[1:]
    return input_seq,target_seq

ds_sequencies=ds_chunk.map(split_input_target)

for exja in ds_sequencies.take(2):
    print('Input (X)',repr(''.join(char_array[exja[0].numpy()])))
    print('Target (y)',repr(''.join(char_array[exja[1].numpy()])))
    print()


# In[50]:


batch_size=64
buffer_size=10000
ds=ds_sequencies.shuffle(buffer_size).batch(batch_size)

def bulid_model(vocab_size,embedding_dim,rnn_unit):
    model=tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim),
        tf.keras.layers.LSTM(rnn_unit,return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

tf.random.set_seed(1)
charset_size=len(char_array)
embedding_dim=256
rnn_unit=512
model=bulid_model(vocab_size=charset_size,embedding_dim=embedding_dim,rnn_unit=rnn_unit)
model.summary()


# In[51]:


model.compile(
    optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(ds,epochs=20)


# In[8]:


# Ejemplo de uso de tf.random.categorical biblioteca. Esta funcion da una busqueda aleatoria de valores referentes a la probabilidad de ocurrencia de los valores logits (segun el valor logits sera proporcional a la cantidad de apariciones)

logits=[[1.0,1.0,1.0]]
print('Probabilities:  ',tf.math.softmax(logits).numpy()[0])


# In[12]:


sample=tf.random.categorical(logits=logits,num_samples=10)
tf.print(sample.numpy())


# In[56]:


logits=[[1.0,1.,3.,]]
print('Probabilities:  ',tf.math.softmax(logits).numpy()[0])


# In[57]:


sample=tf.random.categorical(logits=logits,num_samples=10)
tf.print(sample.numpy())


# In[73]:


def sample(model,starting_str,len_genered_text=500,max_input_length=40,scale_factor=1.0):
    encode_input=[char2int[s] for s in starting_str]
    encode_input=tf.reshape(encode_input,(1,-1))
    generated_str=starting_str
    model.reset_states()
    for i in range(len_genered_text):
        logits=model(encode_input)
        logits=tf.squeeze(logits,0)
        scale_logits=logits*scale_factor
        new_char_indx=tf.random.categorical(scale_logits,num_samples=1)
   
        new_char_indx=tf.squeeze(new_char_indx)[-1].numpy()
      
        generated_str += str(char_array[new_char_indx])

        new_char_indx=tf.expand_dims([new_char_indx],0)
     
        encode_input=tf.concat(
            [encode_input,new_char_indx],axis=1
        )

        encode_input=encode_input[:,-max_input_length:]
    
    return generated_str


# In[74]:


print(sample(model,starting_str='The island',scale_factor=1.0))


# In[60]:


logits=np.array([[1.,1.,3.,]])
print('Probabilities before scaling: ',tf.math.softmax(logits).numpy()[0] )
print('Probabilities after scaling with 0.5: ',tf.math.softmax(0.5*logits).numpy()[0])
print('Probabilities after scaling with 0.1:',tf.math.softmax(0.1*logits).numpy()[0])


# In[67]:


# escalado a 2.0 mas predictibilidad

print(sample(model,starting_str='The island',scale_factor=2.0))


# In[62]:


# escalado a 0.5 mas aleatoriedad
print(sample(model,starting_str='The island',scale_factor=0.5))


# In[84]:


tensor=tf.constant(tf.random.normal(shape=(40,80)))

for i in range(10):
    indx=tf.random.categorical(tensor,num_samples=1)
    print('Indx antes de squeeze: ',indx)
    indx=tf.squeeze(indx)[-1].numpy()
    indx=tf.expand_dims([indx],0)
    print('Indx despues de squeeze: ',indx.numpy())


# In[85]:


indx=tf.random.categorical(tensor,num_samples=1)
tf.print(indx)


# In[86]:


tensor=tf.constant(tf.random.normal(shape=(3,3)))
indx=tf.random.categorical(tensor,num_samples=1)
tf.print(indx)


# In[95]:


array=np.array(np.random.normal(size=(3,3)))
indx=tf.random.categorical(array,num_samples=1)
tf.print(indx)


# In[103]:


ll=[[1.,2.,3.,4.,5.],[2.,6.,4.,6.,5.],[4.,8.,9.,7.,9.]]
indx=tf.random.categorical(ll,num_samples=1)
tf.print(indx)


# In[ ]:




