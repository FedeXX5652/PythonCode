
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 16 - Modelado de datos secuenciales mediante redes neuronales recurrentes
# 
# 

# Note that the optional watermark extension is a small IPython notebook plugin that is being used to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# - [Introducir datos secuenciales](#Introducir-datos-secuenciales)
#   - [Modelar datos secuenciales: el orden sí importa](#Modelar-datos-secuenciales:-el-orden-sí-importa)
#   - [Las diferentes categorías del modelado de secuencias](#Las-diferentes-categorías-del-modelado-de-secuencias)
# - [RNN para modelar secuencias](#RNN-para-modelar-secuencias)
#   - [Entender la estructura y el flujo de una RNN](#Entender-la-estructura-y-el-flujo-de-una-RNN)
#   - [Calcular activaciones en una RNN](#Calcular-activaciones-en-una-RNN)
#   - [Los retos del aprendizaje de interacciones de largo alcance](#Los-retos-del-aprendizaje-de-interacciones-de-largo-alcance)
#   - [Unidades de LSTM](#Unidades-de-LSTM)
# - [Implementar una RNN multicapa para modelar secuencias en TensorFlow](#Implementar-una-RNN-multicapa-para-modelar-secuencias-en-TensorFlow)
#   - [Proyecto uno: crear un análisis de sentimiento de las críticas de películas IMDb con RNN multicapa](#Proyecto-uno---crear-un-análisis-de-sentimiento-de-las-críticas-de-películas-IMDb-con-RNN-multicapa)
#     - [Preparar los datos](#Preparar-los-datos)
#     - [Embedding](#Embedding)
#     - [Construir un modelo de RNN](#Construir-un-modelo-de-RNN)
#       - [Paso 1: definir celdas RNN multicapa](#Paso-1:-definir-celdas-RNN-multicapa)
#       - [Paso 2: definir los estados iniciales para las celdas RNN](#Paso-2:-definir-los-estados-iniciales-para-las-celdas-RNN)
#       - [Paso 3: crear la RNN utilizando las celdas RNN y sus estados](#Paso-3:-crear-la-RNN-utilizando-las-celdas-RNN-y-sus-estados)
#   - [Proyecto dos: implementar una RNN para el modelado de lenguaje a nivel de carácter en TensorFlow](#Proyecto-dos:-implementar-una-RNN-para-el-modelado-de-lenguaje-a-nivel-de-carácter-en-TensorFlow)
#     - [Preparar los datos](#Preparar-los-datos)
#     - [Construir un modelo RNN a nivel de carácter](#Construir-un-modelo-RNN-a-nivel-de-carácter)
# - [Summary](#Summary)

# En[2]:


from IPython.display import Image


# En[ ]:


import gzip


with gzip.open('movie_data.csv.gz') as f_in, open('movie_data.csv', 'wb') as f_out:
    f_out.writelines(f_in)


# # Introducir datos secuenciales 

# ## Modelar datos secuenciales: el orden sí importa

# ## Representar secuencias

# En[3]:




# ## Las diferentes categorías del modelado de secuencias

# En[4]:




# # RNN para modelar secuencias

# ## Entender la estructura y el flujo de una RNN

# En[5]:




# En[6]:




# ## Calcular activaciones en una RNN

# En[7]:




# En[8]:




# ## Los retos del aprendizaje de interacciones de largo alcance

# En[9]:




# ## Unidades de LSTM

# En[10]:




# # Implementar una RNN multicapa para modelar secuencias en TensorFlow

# ## Proyecto uno: crear un análisis de sentimiento de las críticas de películas IMDb con RNN multicapa

# ### Preparar los datos

# En[11]:




# En[1]:


import pyprind
import pandas as pd
from string import punctuation
import re
import numpy as np


df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))


# En[ ]:


## @Lectores: IGNORAD ESTE PÁRRAFO
##
## Este párrafo se incluye para generar más  
## salidas de "registro" cuando este documento se ejecuta  
## en la plataforma Travis Continuous Integration
## para probar el código, así como para 
## acelerar la ejecución mediante un conjunto de datos 
## más pequeño para depurar

import os


if 'TRAVIS' in os.environ:
    df = pd.read_csv('movie_data.csv', encoding='utf-8', nrows=500)


# En[2]:


## Preprocesar los datos:
## Separar palabras y
## contar cada vez que aparece una palabra


from collections import Counter


counts = Counter()
pbar = pyprind.ProgBar(len(df['review']),
                       title='Counting words occurences')
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' '                     for c in review]).lower()
    df.loc[i,'review'] = text
    pbar.update()
    counts.update(text.split())


# En[3]:


## Crear un mapa
## Mapear cada palabra única con un entero

word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}


mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']),
                       title='Map reviews to ints')
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()


# En[4]:


## ## Definir secuencias con la misma longitud
## si la secuencia tiene una longitud < 200: rellenar por la izquierda con ceros
## si la secuencia tiene una longitud > 200: utilizar los últimos 200 elementos

sequence_length = 200  ## longitud de secuencia (Conocida como T en nuestras fórmulas RNN)
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)
for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]

X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].values


np.random.seed(123) # for reproducibility

## Función para generar minilotes:
def create_batch_generator(x, y=None, batch_size=64):
    n_batches = len(x)//batch_size
    x= x[:n_batches*batch_size]
    if y is not None:
        y = y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        if y is not None:
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            yield x[ii:ii+batch_size]


# En[ ]:


## @Lectores: IGNORAD ESTE PÁRRAFO
##
## Este párrafo se incluye para generar más  
## salidas de "registro" cuando este documento se ejecuta  
## en la plataforma Travis Continuous Integration
## para probar el código, así como para 
## acelerar la ejecución mediante un conjunto de datos 
## más pequeño para depurar

if 'TRAVIS' in os.environ:
    X_train = sequences[:250, :]
    y_train = df.loc[:250, 'sentiment'].values
    X_test = sequences[250:500, :]
    y_test = df.loc[250:500, 'sentiment'].values


# ### Embedding

# En[12]:




# ### Construir un modelo de RNN

# En[5]:


import tensorflow as tf


class SentimentRNN(object):
    def __init__(self, n_words, seq_len=200,
                 lstm_size=256, num_layers=1, batch_size=64,
                 learning_rate=0.0001, embed_size=200):
        self.n_words = n_words
        self.seq_len = seq_len
        self.lstm_size = lstm_size   ## número de unidades ocultas
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        ## Definir los marcadores de posición
        tf_x = tf.placeholder(tf.int32,
                    shape=(self.batch_size, self.seq_len),
                    name='tf_x')
        tf_y = tf.placeholder(tf.float32,
                    shape=(self.batch_size),
                    name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32,
                    name='tf_keepprob')
        ## Crear una capa de incrustación
        embedding = tf.Variable(
                    tf.random_uniform(
                        (self.n_words, self.embed_size),
                        minval=-1, maxval=1),
                    name='embedding')
        embed_x = tf.nn.embedding_lookup(
                    embedding, tf_x, 
                    name='embeded_x')

        ## Definir las celdas LSTM y mantenerlas juntas
        cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                   tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
                   output_keep_prob=tf_keepprob)
                 for i in range(self.num_layers)])

        ## Definir el estado inicial:
        self.initial_state = cells.zero_state(
                 self.batch_size, tf.float32)
        print('  << initial state >> ', self.initial_state)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                 cells, embed_x,
                 initial_state=self.initial_state)
        ## Nota: lstm_outputs shape: 
        ##  [batch_size, max_time, cells.output_size]
        print('\n  << lstm_output   >> ', lstm_outputs)
        print('\n  << final state   >> ', self.final_state)

        ## Aplicar una capa FC después por encima de la salida de RNN:
        logits = tf.layers.dense(
                 inputs=lstm_outputs[:, -1],
                 units=1, activation=None,
                 name='logits')
        
        logits = tf.squeeze(logits, name='logits_squeezed')
        print ('\n  << logits        >> ', logits)
        
        y_proba = tf.nn.sigmoid(logits, name='probabilities')
        predictions = {
            'probabilities': y_proba,
            'labels' : tf.cast(tf.round(y_proba), tf.int32,
                 name='labels')
        }
        print('\n  << predictions   >> ', predictions)

        ## Definir la función de coste
        cost = tf.reduce_mean(
                 tf.nn.sigmoid_cross_entropy_with_logits(
                 labels=tf_y, logits=logits),
                 name='cost')
        
        ## Definir el optimizador
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

    def train(self, X_train, y_train, num_epochs):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                state = sess.run(self.initial_state)
                
                for batch_x, batch_y in create_batch_generator(
                            X_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keepprob:0': 0.5,
                            self.initial_state : state}
                    loss, _, state = sess.run(
                            ['cost:0', 'train_op', 
                             self.final_state],
                            feed_dict=feed)

                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d "
                              "| Train loss: %.5f" % (
                               epoch + 1, num_epochs,
                               iteration, loss))

                    iteration +=1
                if (epoch+1)%10 == 0:
                    self.saver.save(sess,
                        "model/sentiment-%d.ckpt" % epoch)

    def predict(self, X_data, return_proba=False):
        preds = []
        with tf.Session(graph = self.g) as sess:
            self.saver.restore(
                sess, tf.train.latest_checkpoint('model/'))
            test_state = sess.run(self.initial_state)
            for ii, batch_x in enumerate(
                create_batch_generator(
                    X_data, None, batch_size=self.batch_size), 1):
                feed = {'tf_x:0' : batch_x,
                        'tf_keepprob:0': 1.0,
                        self.initial_state : test_state}
                if return_proba:
                    pred, test_state = sess.run(
                        ['probabilities:0', self.final_state],
                        feed_dict=feed)
                else:
                    pred, test_state = sess.run(
                        ['labels:0', self.final_state],
                        feed_dict=feed)
                    
                preds.append(pred)
                
        return np.concatenate(preds)


# #### Paso 1: definir celdas RNN multicapa

# #### Paso 2: definir los estados iniciales para las celdas RNN
# 

# #### Paso 3: crear la RNN utilizando las celdas RNN y sus estados
# 
# 

# En[6]:


## Entrenamiento:

n_words = max(list(word_to_int.values())) + 1

rnn = SentimentRNN(n_words=n_words, 
                   seq_len=sequence_length,
                   embed_size=256, 
                   lstm_size=128, 
                   num_layers=1, 
                   batch_size=100, 
                   learning_rate=0.001)


# En[7]:


rnn.train(X_train, y_train, num_epochs=40)


# En[ ]:


## Prueba: 
preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: %.3f' % (
      np.sum(preds == y_true) / len(y_true)))


# En[ ]:


## Obtener probabilidades:
proba = rnn.predict(X_test, return_proba=True)


# ## Proyecto dos: implementar una RNN para el modelado de lenguaje a nivel de carácter en TensorFlow

# En[13]:




# ### Preparar los datos
# 

# En[14]:




# En[15]:




# En[16]:




# En[1]:


import numpy as np


## Leer y procesar el texto
with open('pg2265.txt', 'r', encoding='utf-8') as f: 
    text=f.read()

text = text[15858:]
chars = set(text)
char2int = {ch:i for i,ch in enumerate(chars)}
int2char = dict(enumerate(chars))
text_ints = np.array([char2int[ch] for ch in text], 
                     dtype=np.int32)


# En[ ]:


## @Lectores: IGNORAD ESTE PÁRRAFO
##
## Este párrafo se incluye para generar más  
## salidas de "registro" cuando este documento se ejecuta  
## en la plataforma Travis Continuous Integration
## para probar el código, así como para 
## acelerar la ejecución mediante un conjunto de datos 
## más pequeño para depurar

if 'TRAVIS' in os.environ:
    text = text[:1000]
    chars = set(text)
    char2int = {ch:i for i,ch in enumerate(chars)}
    int2char = dict(enumerate(chars))
    text_ints = np.array([char2int[ch] for ch in text], 
                         dtype=np.int32)


# En[2]:


def reshape_data(sequence, batch_size, num_steps):
    tot_batch_length = batch_size * num_steps
    num_batches = int(len(sequence) / tot_batch_length)
    if num_batches*tot_batch_length + 1 > len(sequence):
        num_batches = num_batches - 1
    ## Truncar la secuencia por el final para deshacernos de 
    ## los caracteres restantes que no generan un lote completo
    x = sequence[0 : num_batches*tot_batch_length]
    y = sequence[1 : num_batches*tot_batch_length + 1]
    ## Dividir x & y en una lista de lotes de secuencias: 
    x_batch_splits = np.split(x, batch_size)
    y_batch_splits = np.split(y, batch_size)
    ## Mantener los lotes juntos
    ## batch_size x tot_batch_length
    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)
    
    return x, y

## Probar:
train_x, train_y = reshape_data(text_ints, 64, 10)
print(train_x.shape)
print(train_x[0, :10])
print(train_y[0, :10])
print(''.join(int2char[i] for i in train_x[0, :50]))


# En[3]:


np.random.seed(123)

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, tot_batch_length = data_x.shape    
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:, b*num_steps: (b+1)*num_steps], 
               data_y[:, b*num_steps: (b+1)*num_steps])
        
bgen = create_batch_generator(train_x[:,:100], train_y[:,:100], 15)
for b in bgen:
    print(b[0].shape, b[1].shape, end='  ')
    print(''.join(int2char[i] for i in b[0][0,:]).replace('\n', '*'), '    ',
          ''.join(int2char[i] for i in b[1][0,:]).replace('\n', '*'))


# ### Construir un modelo RNN a nivel de carácter

# En[4]:


import tensorflow as tf
import os

class CharRNN(object):
    def __init__(self, num_classes, batch_size=64, 
                 num_steps=100, lstm_size=128, 
                 num_layers=1, learning_rate=0.001, 
                 keep_prob=0.5, grad_clip=5, 
                 sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)

            self.build(sampling=sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            
    def build(self, sampling):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps

        tf_x = tf.placeholder(tf.int32, 
                              shape=[batch_size, num_steps], 
                              name='tf_x')
        tf_y = tf.placeholder(tf.int32, 
                              shape=[batch_size, num_steps], 
                              name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32, 
                              name='tf_keepprob')

        # Codificación one-hot:
        x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth=self.num_classes)

        ### Construir las celdas RNN multicapa
        cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(self.lstm_size), 
                output_keep_prob=tf_keepprob) 
            for _ in range(self.num_layers)])
        
        ## Definir el estado inicial
        self.initial_state = cells.zero_state(
                    batch_size, tf.float32)

        ## Ejecutar cada secuencia a través de la RNN 
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                    cells, x_onehot, 
                    initial_state=self.initial_state)
        
        print('  << lstm_outputs  >>', lstm_outputs)

        seq_output_reshaped = tf.reshape(
                    lstm_outputs, 
                    shape=[-1, self.lstm_size],
                    name='seq_output_reshaped')

        logits = tf.layers.dense(
                    inputs=seq_output_reshaped, 
                    units=self.num_classes,
                    activation=None,
                    name='logits')

        proba = tf.nn.softmax(
                    logits, 
                    name='probabilities')
        print(proba)

        y_reshaped = tf.reshape(
                    y_onehot, 
                    shape=[-1, self.num_classes],
                    name='y_reshaped')
        cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, 
                        labels=y_reshaped),
                    name='cost')

        # Recortar el gradiente para evitar una "explosión del gradiente"
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
                    tf.gradients(cost, tvars), 
                    self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    name='train_op')
        
    def train(self, train_x, train_y, 
              num_epochs, ckpt_dir='./model/'):
        ## Crear el directorio de control
		## si este no existe
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)

            n_batches = int(train_x.shape[1]/self.num_steps)
            iterations = n_batches * num_epochs
            for epoch in range(num_epochs):

                # Entrenar la red
                new_state = sess.run(self.initial_state)
                loss = 0
                ## Generador de minilotes:
                bgen = create_batch_generator(
                        train_x, train_y, self.num_steps)
                for b, (batch_x, batch_y) in enumerate(bgen, 1):
                    iteration = epoch*n_batches + b
                    
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keepprob:0': self.keep_prob,
                            self.initial_state : new_state}
                    batch_cost, _, new_state = sess.run(
                            ['cost:0', 'train_op', 
                                self.final_state],
                            feed_dict=feed)
                    if iteration % 10 == 0:
                        print('Epoch %d/%d Iteration %d'
                              '| Training loss: %.4f' % (
                              epoch + 1, num_epochs, 
                              iteration, batch_cost))

                ## Guardar el modelo entrenado    
                self.saver.save(
                        sess, os.path.join(
                            ckpt_dir, 'language_modeling.ckpt'))
                              
                              
                
    def sample(self, output_length, 
               ckpt_dir, starter_seq="The "):
        observed_seq = [ch for ch in starter_seq]        
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(
                sess, 
                tf.train.latest_checkpoint(ckpt_dir))
            ## 1: ejecutar el modelo con la secuencia de inicio
            new_state = sess.run(self.initial_state)
            for ch in starter_seq:
                x = np.zeros((1, 1))
                x[0,0] = char2int[ch]
                feed = {'tf_x:0': x,
                        'tf_keepprob:0': 1.0,
                        self.initial_state: new_state}
                proba, new_state = sess.run(
                        ['probabilities:0', self.final_state], 
                        feed_dict=feed)

            ch_id = get_top_char(proba, len(chars))
            observed_seq.append(int2char[ch_id])
            
            ## 2: ejecutar el modelo con el observed_seq actualizado
            for i in range(output_length):
                x[0,0] = ch_id
                feed = {'tf_x:0': x,
                        'tf_keepprob:0': 1.0,
                        self.initial_state: new_state}
                proba, new_state = sess.run(
                        ['probabilities:0', self.final_state], 
                        feed_dict=feed)

                ch_id = get_top_char(proba, len(chars))
                observed_seq.append(int2char[ch_id])

        return ''.join(observed_seq)


# En[5]:


def get_top_char(probas, char_size, top_n=5):
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p / np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id


# En[6]:


batch_size = 64
num_steps = 100 
train_x, train_y = reshape_data(text_ints, 
                                batch_size, 
                                num_steps)

rnn = CharRNN(num_classes=len(chars), batch_size=batch_size)
rnn.train(train_x, train_y, 
          num_epochs=100,
          ckpt_dir='./model-100/')


# En[7]:


np.random.seed(123)
rnn = CharRNN(len(chars), sampling=True)

print(rnn.sample(ckpt_dir='./model-100/', 
                 output_length=500))


# En[8]:


## ejecutar durante 200 épocas
batch_size = 64
num_steps = 100 

rnn = CharRNN(num_classes=len(chars), batch_size=batch_size)
rnn.train(train_x, train_y, 
          num_epochs=200,
          ckpt_dir='./model-200/')


# En[9]:


del rnn

np.random.seed(123)
rnn = CharRNN(len(chars), sampling=True)
print(rnn.sample(ckpt_dir='./model-200/', 
                 output_length=500))


# # Resumen

# ...

# ---
# 
# Los lectores deberán ignorar la siguiente celda.

# En[ ]:



