
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# # Capítulo 14 - Ir más lejos: la mecánica de TensorFlow

# - [Características clave de TensorFlow](#Características-clave-de-TensorFlow)
# - [Entender los grafos computacionales de TensorFlow](#Entender-los-grafos-computacionales-de-TensorFlow)
#   - [Marcadores de posición en TensorFlow](#Marcadores-de-posición-en-TensorFlow)
#     - [Definir marcadores de posición](#Definir-marcadores-de-posición)
#     - [Alimentar marcadores de posición con datos](#Alimentar-marcadores-de-posición-con-datos)
#     - [Definir marcadores de posición para matrices de datos con diferentes tamaños de lote](#Definir-marcadores-de-posición-para-matrices-de-datos-con-diferentes-tamaños-de-lote)
#   - [Variables en TensorFlow](#Variables-en-TensorFlow)
#     - [Definir variables](#Definir-variables)
#     - [Inicializar variables](#Inicializar-variables)
#     - [Alcance de la variable](#Alcance-de-la-variable)
#     - [Reutilizar variables](#Reutilizar-variables)
#   - [Crear un modelo de regresión](#Crear-un-modelo-de-regresión)
#   - [Ejecutar objetos en un grafo de TensorFlow mediante sus nombres](#Ejecutar-objetos-en-un-grafo-de-TensorFlow-mediante-sus-nombres)
# - [Almacenar y restablecer un modelo en TensorFlow](#Almacenar-y-restablecer-un-modelo-en-TensorFlow)
# - [Transformar tensores como matrices de datos multidimensionales](#Transformar-tensores-como-matrices-de-datos-multidimensionales)
# - [Utilizar la mecánica de control de flujo para crear grafos](#Utilizar-la-mecánica-de-control-de-flujo-para-crear-grafos)
# - [Visualizar el grafo con TensorBoard](#Visualizar-el-grafo-con-TensorBoard)
# - [Resumen](#Resumen)

# En[2]:


from IPython.display import Image


# En[3]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# ## Características clave de TensorFlow

# En[4]:




# **Cómo obtener el rango y la forma de un tensor**

# En[5]:


g = tf.Graph()

## definir el gráfico de cálculo
with g.as_default():
    ## definir tensores t1, t2, t3:
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])

    ## obtener sus rangos
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)

    ## obtener sus formas
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print('Shapes:', s1, s2, s3)

with tf.Session(graph=g) as sess:
    print('Ranks:', 
          r1.eval(), 
          r2.eval(), 
          r3.eval())


# ## Entender los grafos computacionales de TensorFlow

# En[6]:


Image("images/14_02.png")


# En[7]:


g = tf.Graph()
 
## añadir nodos al grafo
with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b') 
    c = tf.constant(3, name='c') 

    t = 2*(a-b) + c
    
## iniciar el grafo
with tf.Session(graph=g) as sess:
    print('2*(a-b)+c => ', sess.run(t))


# ## Marcadores de posición en TensorFlow

# #### Definir marcadores de posición

# En[8]:


g = tf.Graph()
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[],
                          name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[],
                          name='tf_b') 
    tf_c = tf.placeholder(tf.int32, shape=[],
                          name='tf_c') 

    r1 = tf_a-tf_b
    r2 = 2*r1
    z  = r2 + tf_c


# #### Alimentar marcadores de posición con datos

# En[9]:


## iniciar el grafo anterior
with tf.Session(graph=g) as sess:
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('z:', 
          sess.run(z, feed_dict=feed))


# Ejecución con y sin alimentar tf_c:

# En[10]:


## iniciar el grafo anterior
with tf.Session(graph=g) as sess:
    ## ejecución sin alimentar tf_c
    feed = {tf_a: 1,
            tf_b: 2}
    print('r1:', 
          sess.run(r1, feed_dict=feed))
    print('r2:', 
          sess.run(r2, feed_dict=feed))
    
    ## ejecución alimentando tf_c
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('r1:', 
          sess.run(r1, feed_dict=feed))
    print('r2:', 
          sess.run(r2, feed_dict=feed))


# ### Definir marcadores de posición para matrices de datos con diferentes tamaños de lote

# Marcadores de posición para distintos tamaños de lote:

# En[11]:


g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(tf.float32, 
                          shape=[None, 2],
                          name='tf_x')
    
    x_mean = tf.reduce_mean(tf_x, 
                          axis=0, 
                          name='mean')


np.random.seed(123)
np.set_printoptions(precision=2)

with tf.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, 
                           size=(5,2))
    print('Feeding data with shape', x1.shape)
    print('Result:', sess.run(x_mean, 
                             feed_dict={tf_x:x1}))
    x2 = np.random.uniform(low=0, high=1, 
                           size=(10,2))
    print('Feeding data with shape', x2.shape)
    print('Result:', sess.run(x_mean, 
                             feed_dict={tf_x:x2}))


# En[12]:


print(tf_x)


# ### Variables en TensorFlow
# 

# #### Definir variables

# En[13]:


g1 = tf.Graph()

with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]), name='w')
    print(w)


# #### Inicializar variables

# En[14]:


## inicializar w y evaluarlo
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))


# En[15]:


## añadir el init_op al grafo
with g1.as_default():
    init_op = tf.global_variables_initializer()
    
## inicializar w con init_op y evaluarlo
with tf.Session(graph=g1) as sess:
    sess.run(init_op)
    print(sess.run(w))


# En[16]:


g2 = tf.Graph()

with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(2, name='w2')


# En[17]:


with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print('w1:', sess.run(w1))


# Error si una variable no se inicializa:

# En[18]:


with tf.Session(graph=g2) as sess:
    
    try:
        sess.run(init_op)
        print('w2:', sess.run(w2))
    except tf.errors.FailedPreconditionError as e:
        print(e)


# #### Alcance de la variable

# En[19]:


g = tf.Graph()

with g.as_default():
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer-1'):
            w1 = tf.Variable(tf.random_normal(
                shape=(10,4)), name='weights')
        with tf.variable_scope('layer-2'):
            w2 = tf.Variable(tf.random_normal(
                shape=(20,10)), name='weights')
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer-1'):
            w3 = tf.Variable(tf.random_normal(
                shape=(10,4)), name='weights')

    print(w1)
    print(w2)
    print(w3)


# #### Reutilizar variables

# En[20]:


######################
## Funciones ayudante ##
######################

def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights',
                              shape=(data_shape[1], n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias', 
                           initializer=tf.zeros(shape=n_classes))
    print(weights)
    print(bias)
    logits = tf.add(tf.matmul(data, weights), 
                    bias, 
                    name='logits')
    print(logits)
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(
        tf.random_normal(shape=(data_shape[1], 
                                n_hidden)),
        name='w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden),
                     name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, 
                    name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
        
    w2 = tf.Variable(
        tf.random_normal(shape=(n_hidden, 
                                data_shape[1])),
        name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),
                     name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2, 
                    name = 'output')
    return output, tf.nn.sigmoid(output)
    
########################
## Definir el grafo ##
########################

batch_size=64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), 
                          dtype=tf.float32,
                          name='tf_X')
    ## crear el generador
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, 
                                   n_hidden=50)
    
    ## crear el clasificador
    with tf.variable_scope('classifier') as scope:
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, 
                                    labels=tf.ones(
                                        shape=batch_size))
        
        ## reutilizar el clasificador para datos generados
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(
                                        shape=batch_size))
        
        init_op = tf.global_variables_initializer()


# En[21]:


## modo alternativo

g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), 
                          dtype=tf.float32,
                          name='tf_X')
    ## crear el generador
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, 
                                   n_hidden=50)
    
    ## crear el clasificador
    with tf.variable_scope('classifier'):
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, 
                                    labels=tf.ones(
                                        shape=batch_size))
        
    with tf.variable_scope('classifier', reuse=True):
        ## reutilizar el clasificador para datos generados 
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(
                                        shape=batch_size))
        
        init_op = tf.global_variables_initializer()


# ### Crear un modelo de regresión

# En[22]:


## definir un grafo
g = tf.Graph()

## definir el grafo computacional
with g.as_default():
    ## texto por defecto
    tf.set_random_seed(123)
    tf_x = tf.placeholder(shape=(None), 
                          dtype=tf.float32, 
                          name='tf_x')
    tf_y = tf.placeholder(shape=(None), 
                          dtype=tf.float32,
                          name='tf_y')
    
    ## definir la variable (parámetros del modelo)
    weight = tf.Variable(
        tf.random_normal(
            shape=(1, 1), 
            stddev=0.25),
        name = 'weight')
    bias = tf.Variable(0.0, name='bias')
    
    ## crear el modelo
    y_hat = tf.add(weight * tf_x, bias, 
                   name='y_hat')
    print(y_hat)
    
    ## calcular el coste
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), 
                          name='cost')
    print(cost)
    
    ## entrenar
    optim = tf.train.GradientDescentOptimizer(
        learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')


# En[23]:


## crear un conjunto de datos aleatorio para regresión


np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, 
                             scale=(0.5 + t*t/3), 
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)


x, y = make_random_data() 

plt.plot(x, y, 'o')
plt.savefig('images/14_03.png', dpi=300)
plt.show()


# ### Ejecutar objetos en un grafo de TensorFlow mediante sus nombres

# En[24]:


## dividir entrenamiento/prueba:
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]


## entrenar el modelo
n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    ## primero, ejecutar el inicializador de variables
    sess.run(tf.global_variables_initializer())
    
    ## entrenar el modelo durante n_épocas
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], 
                        feed_dict={tf_x: x_train,
                                   tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))
            

plt.plot(training_costs)
plt.savefig('images/14_04.png', dpi=300)


# Ejecutar con nombres de variables:

# En[25]:


## dividir entrenamiento/prueba
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]

## representar los datos entrenados
plt.plot(x_train, y_train, 'o')
plt.show()

## entrenar el modelo
n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    ## primero, ejecutar el inicializador de variables
    sess.run(tf.global_variables_initializer())
    
    ## entrenar el modelo durante n_épocas
    for e in range(n_epochs):
        c, _ = sess.run(['cost:0', 'train_op'], 
                        feed_dict={'tf_x:0': x_train,
                                   'tf_y:0': y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))


# ## Almacenar y restablecer un modelo en TensorFlow

# En[26]:


## añadir un saver al grafo
with g.as_default():
    saver = tf.train.Saver()
    
## entrenar el modelo
n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    ## primero, ejecutar el inicializador de variables
    sess.run(tf.global_variables_initializer())
    
    ## entrenar el modelo durante n_épocas
    for e in range(n_epochs):
        c, _ = sess.run(['cost:0', 'train_op'], 
                        feed_dict={'tf_x:0':x_train,
                                   'tf_y:0':y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))
            
    saver.save(sess, './trained-model')


# Restablecer el modelo guardado:

# En[27]:


## nuevo archivo: cargar un modelo entrenado
## y ejecutar el modelo en el conjunto de prueba


g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_pred = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_test})


# En[28]:


print('SSE: %.4f' % (np.sum(np.square(y_pred - y_test))))


# En[29]:


x_arr = np.arange(-2, 4, 0.1)

g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_arr = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_arr})

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr.T[:, 0], '-r', lw=3)
plt.savefig('images/14_05.png', dpi=400)
plt.show()


# ## Transformar tensores como matrices de datos multidimensionales

# En[30]:


g = tf.Graph()
with g.as_default():
    arr = np.array([[1., 2., 3., 3.5],
                    [4., 5., 6., 6.5],
                    [7., 8., 9., 9.5]])
    T1 = tf.constant(arr, name='T1')
    print(T1)
    s = T1.get_shape()
    print('Shape of T1 is', s)
    T2 = tf.Variable(tf.random_normal(
        shape=s))
    print(T2)
    T3 = tf.Variable(tf.random_normal(
        shape=(s.as_list()[0],)))
    print(T3)


# En[31]:


with g.as_default():
    T4 = tf.reshape(T1, shape=[1, 1, -1], 
                    name='T4')
    print(T4)
    T5 = tf.reshape(T1, shape=[1, 3, -1], 
                    name='T5')
    print(T5)


# En[32]:


with tf.Session(graph = g) as sess:
    print(sess.run(T4)) 
    print()   
    print(sess.run(T5))


# En[33]:


with g.as_default():
    T6 = tf.transpose(T5, perm=[2, 1, 0], 
                     name='T6')
    print(T6)
    T7 = tf.transpose(T5, perm=[0, 2, 1], 
                     name='T7')
    print(T7)


# En[34]:


with g.as_default():
    t5_splt = tf.split(T5, 
                       num_or_size_splits=2, 
                       axis=2, name='T8')
    print(t5_splt)


# En[35]:


g = tf.Graph()
with g.as_default():
    t1 = tf.ones(shape=(10, 1), 
                 dtype=tf.float32, name='t1')
    t2 = tf.zeros(shape=(10, 1),
                 dtype=tf.float32, name='t1')
    print(t1)
    print(t2)
    
with g.as_default():
    t3 = tf.concat([t1, t2], axis=0, name='t3')
    print(t3)
    t4 = tf.concat([t1, t2], axis=1, name='t4')
    print(t4)


# En[36]:


with tf.Session(graph = g) as sess:
    print(t3.eval())
    print()
    print(t4.eval())


# ## Utilizar la mecánica de control de flujo para crear grafos

# En[37]:


## control de flujo en Python 


x, y = 1.0, 2.0

g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_y')
    if x < y:
        res = tf.add(tf_x, tf_y, name='result_add')
    else:
        res = tf.subtract(tf_x, tf_y, name='result_sub')
        
    print('Object: ', res)
        
with tf.Session(graph=g) as sess:
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x, 
                              'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x,
                              'tf_y:0': y}))  
    
    ## descomenta la línea siguiente si deseas visualizar el grafo en TensorBoard:
    file_writer = tf.summary.FileWriter(logdir='./logs/py-cflow/', graph=g)


# En[38]:


## control de flujo en TensorFlow 


x, y = 1.0, 2.0

g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_y')
    res = tf.cond(tf_x < tf_y, 
                  lambda: tf.add(tf_x, tf_y, 
                                 name='result_add'),
                  lambda: tf.subtract(tf_x, tf_y, 
                                 name='result_sub'))
    print('Object:', res)
        
with tf.Session(graph=g) as sess:
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x, 
                              'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x,
                              'tf_y:0': y}))  

    #file_writer = tf.summary.FileWriter(logdir='./logs/tf-cond/', graph=g)


# En[39]:




# ## ViVisualizar el grafo con TensorBoard

# En[40]:


import tensorflow as tf


###########################
##   Funciones ayudante    ##
###########################

def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name = 'weights',
                              shape=(data_shape[1],
                                     n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias', 
                           initializer=tf.zeros(
                                     shape=n_classes))
    print(weights)
    print(bias)
    logits = tf.add(tf.matmul(data, weights), 
                    bias, 
                    name='logits')
    print(logits)
    return logits, tf.nn.softmax(logits)


def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(
        tf.random_normal(shape=(data_shape[1], 
                                n_hidden)),
        name='w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden),
                     name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, 
                    name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
        
    w2 = tf.Variable(
        tf.random_normal(shape=(n_hidden, 
                                data_shape[1])),
        name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),
                     name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2, 
                    name = 'output')
    return output, tf.nn.sigmoid(output)


###########################
##  Crear el grafo   ##
###########################

batch_size=64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), 
                          dtype=tf.float32,
                          name='tf_X')
    
    ## crear el generador
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, 
                                   n_hidden=50)
    
    ## crear el clasificador
    with tf.variable_scope('classifier') as scope:
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, 
                                    labels=tf.ones(
                                        shape=batch_size))
        
        ## reutilizar el clasificador para los datos generados
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(
                                        shape=batch_size))

        
with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    
    file_writer = tf.summary.FileWriter(logdir='logs/', graph=g)



# En[41]:




# En[42]:




# En[43]:




# ## Resumen

# ...

# ---
# 
# Los lectores deberán ignorar la siguiente celda.

# En[]:




# En[]:




