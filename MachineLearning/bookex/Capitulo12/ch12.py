
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 12 - Implementar una red neuronal artificial multicapa desde cero
# 

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# ### Sumario

# - [Modelar funciones complejas con redes neuronales artificiales](#Modelar-funciones-complejas-con-redes-neuronales-artificiales)
#   - [Resumen de una red neuronal de una capa](#Resumen-de-una-red-neuronal-de-una-capa)
#   - [Presentar la arquitectura de red neuronal multicapa](#Presentar-la-arquitectura-de-red-neuronal-multicapa)
#   - [Activar una red neuronal mediante la propagación hacia delante](#Activar-una-red-neuronal-mediante-la-propagación-hacia-delante)
# - [Clasificar dígitos manuscritos](#Clasificar-dígitos-manuscritos)
#   - [Obtener el conjunto de datos MNIST](#Obtener-el-conjunto-de-datos-MNIST)
#   - [Implementar un perceptrón multicapa](#Implementar-un-perceptrón-multicapa)
# - [Entrenar una red neuronal artificial](#Entrenar-una-red-neuronal-artificial)
#   - [Calcular la función de coste logística](#Calcular-la-función-de-coste-logística)
#   - [Desarrollar tu intuición para la propagación hacia atrás](#Desarrollar-tu-intuición-para-la-propagación-hacia-atrás)
#   - [Entrenar redes neuronales mediante la propagación hacia atrás](#Entrenar-redes-neuronales-mediante-la-propagación-hacia-atrás)
# - [Sobre la convergencia en redes neuronales](#Sobre-la-convergencia-en-redes-neuronales)
# - [Resumen](#Resumen)


# En[2]:


from IPython.display import Image


# # Modelar funciones complejas con redes neuronales artificiales

# ...

# ## Resumen de una red neuronal de una capa

# En[3]:





# ## Presentar la arquitectura de red neuronal multicapa

# En[4]:




# En[5]:





# ## Activar una red neuronal mediante la propagación hacia delante

# En[6]:





# # Clasificar dígitos manuscritos

# ...

# ## Obtener el conjunto de datos MNIST

# El conjunto de datos MNIST está disponible públicamente en http://yann.lecun.com/exdb/mnist/ y consta de las siguientes cuatro partes:
# 
# - Conjunto de entrenamiento de imágenes: train-images-idx3-ubyte.gz (9.9 MB, 47 MB sin comprimir, 60.000 muestras)
# - Conjunto de entrenamiento de etiquetas: train-labels-idx1-ubyte.gz (29 KB, 60 KB sin comprimir, 60.000 muestras)
# - Conjunto de prueba de imágenes: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, sin comprimir, 10.000 muestras)
# - Conjunto de prueba de etiquetas: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB sin comprimir, 10.000 etiquetas)
# 
# En esta sección, trabajaremos solo con un subconjunto de MNIST, por lo que solo necesitamos descargar el conjunto de entrenamiento de imágenes y el conjunto de entrenamiento de etiquetas. Una vez descargados los archivos, recomiendo que los descomprimas con la herramienta gzip de Unix/Linux desde la terminal por comodidad, es decir, utilizando el comando  
# 
#     gzip *ubyte.gz -d
#  
# en tu directorio de descargas MNIST local o utilizando tu herramienta para descomprimir favorita si estás trabajando con una máquina que trabaja sobre Microsoft Windows. Las imágenes están almacenadas en formato byte y, con la siguiente función, las leeremos en las matrices de NumPy que utilizaremos para entrenar nuestro MLP.
# 

# En[7]:


import os
import struct
import numpy as np
 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels


# En[8]:




# En[9]:


# descomprimir mnist

import sys
import gzip
import shutil

if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 


# En[10]:


X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))


# En[11]:


X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


# Visualizar el primer dígito de cada clase:

# En[12]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_5.png', dpi=300)
plt.show()


# Visualizar 25 versiones distintas de "7":

# En[13]:


fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_6.png', dpi=300)
plt.show()


# En[14]:


import numpy as np

np.savez_compressed('mnist_scaled.npz', 
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)


# En[15]:


mnist = np.load('mnist_scaled.npz')
mnist.files


# En[16]:


X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 
                                    'X_test', 'y_test']]

del mnist

X_train.shape



# ## Implementar un perceptrón multicapa

# En[17]:


import numpy as np
import sys


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatche_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # paso 1: entrada de red de la capa oculta
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # paso 2: activación de la capa oculta
        a_h = self._sigmoid(z_h)

        # paso 3: entrada de red de la capa de salida
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # paso 4: activación de la capa de salida
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Inicialización del peso
        ########################

        # pesos para entrada -> oculta
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # pesos para oculta -> salida
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterar sobre épocas de entrenamiento
        for i in range(self.epochs):

            # iterar sobre minilotes
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # propagación hacia delante
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Propagación hacia atrás
                ##################

                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularización y actualizaciones de peso
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluación
            #############

            # Evaluación después de cada época durante el entrenamiento
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


# En[18]:


n_epochs = 200

@Lectores: IGNORAD ESTE PÁRRAFO
##
## Este párrafo se incluye para generar más  
## salidas de "registro" cuando este documento se ejecuta  
## en la plataforma Travis Continuous Integration
## para probar el código, así como para 
## acelerar la ejecución mediante un conjunto de datos 
## más pequeño para depurar

if 'TRAVIS' in os.environ:
    n_epochs = 20


# En[19]:


nn = NeuralNetMLP(n_hidden=100, 
                  l2=0.01, 
                  epochs=n_epochs, 
                  eta=0.0005,
                  minibatch_size=100, 
                  shuffle=True,
                  seed=1)

nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])


# ---
# **Nota**
# 
# En el método de definición del ejemplo de MLP anterior,
# 
# ```python
# 
# for idx in mini:
# ...
#     # calcular el gradiente mediante propagación hacia atrás 
#     grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
#                                       a3=a3, z2=z2,
#                                       y_enc=y_enc[:, idx],
#                                       w1=self.w1,
#                                       w2=self.w2)
# 
#     delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
#     self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
#     self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
#     delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
# ```
# 
# `delta_w1_prev` (same applies to `delta_w2_prev`) is a memory view on `delta_w1` via  
# 
# ```python
# delta_w1_prev = delta_w1
# ```
# en la última línea. Esto podría ser un problema, puesto que actualizar `delta_w1 = self.eta * grad1` cambiaría `delta_w1_prev` así como cuando interamos sobre el bucle for. Ten en cuenta que este no es el caso, porque asignamos una nueva matriz a `delta_w1` en cada iteración -- la matriz de gradiente por la tasa de aprendizaje:
# 
# ```python
# delta_w1 = self.eta * grad1
# ```
# 
# La sentencia que acabamos de mostrar deja el `delta_w1_prev` apuntando a la "antigua" matriz `delta_w1`. Para ilustrarlo con un sencillo fragmento de código, puedes tener en cuenta el siguiente ejemplo:
# 
# 

# En[20]:


import numpy as np

a = np.arange(5)
b = a
print('a & b', np.may_share_memory(a, b))


a = np.arange(5)
print('a & b', np.may_share_memory(a, b))


# (Fin de la nota.)
# 
# ---

# En[21]:


import matplotlib.pyplot as plt

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()


# En[22]:


plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.savefig('images/12_08.png', dpi=300)
plt.show()


# En[23]:


y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))


# En[24]:


miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('images/12_09.png', dpi=300)
plt.show()



# # Entrenar una red neuronal artificial

# ...

# ## Calcular la función de coste logística

# En[25]:





# ## Desarrollar tu intuición para la propagación hacia atrás

# ...

# ## Entrenar redes neuronales mediante la propagación hacia atrás

# En[26]:




# En[6]:





# # Sobre la convergencia en redes neuronales

# En[28]:





# ...

# # Resumen

# ...
