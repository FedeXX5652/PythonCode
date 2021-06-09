
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 5 - Comprimir datos mediante la reducción de dimensionalidad

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Sumario

# - [Reducción de dimensionalidad sin supervisión mediante el análisis de componentes principales](#Reducción-de-dimensionalidad-sin-supervisión-mediante-el-análisis-de-componentes-principales)
#   - [Los pasos esenciales que se esconden detrás del análisis de componentes principales](#Los-pasos-esenciales-que-se-esconden-detrás-del-análisis-de-componentes-principales)
#   - [Extraer el componente principal paso a paso](#Extraer-el-componente-principal-paso-a-paso)
#   - [Varianza total y explicada](#Varianza-total-y-explicada)
#   - [Transformación de características](#Transformación-de-características)
#   - [Análisis de componentes principales en scikit-learn](#Análisis-de-componentes-principales-en-scikit-learn)
# - [Compresión de datos supervisada mediante análisis discriminante lineal](#Compresión-de-datos-supervisada-mediante-análisis-discriminante-lineal)
#   - [Análisis de componentes principales frente a análisis discriminante lineal](#Análisis-de-componentes-principales-frente-a-análisis-discriminante-lineal)
#   - [Cómo funciona interiormente el análisis discriminante lineal](#Cómo-funciona-interiormente-el-análisis-discriminante-lineal)
#   - [Calcular las matrices de dispersión](#Calcular-las-matrices-de-dispersión)
#   - [Seleccionar discriminantes lineales para el nuevo subespacio de características](#Seleccionar-discriminantes-lineales-para-el-nuevo-subespacio-de-características)
#   - [Proyectar muestras en el nuevo espacio de características](#Proyectar-muestras-en-el-nuevo-espacio-de-características)
#   - [ADL con scikit-learn](#ADL-con-scikit-learn)
# - [Utilizar el análisis de componentes principales con kernels para mapeos no lineales](#Utilizar-el-análisis-de-componentes-principales-con-kernels-para-mapeos-no-lineales)
#   - [Funciones kernel y el truco del kernel](#Funciones-kernel-y-el-truco-del-kernel)
#   - [Implementar un análisis de componentes principales con kernels en Python](#Implementar-un-análisis-de-componentes-principales-con-kernels-en-Python)
#     - [Ejemplo 1: separar formas de media luna](#Ejemplo 1:-separar-formas-de-media-luna)
#     - [Ejemplo 2: separar círculos concéntricos](#Ejemplo 2: separar círculos concéntricos)
#   - [Proyectar nuevos puntos de datos](#Proyectar-nuevos-puntos-de-datos)
#   - [Análisis de componentes principales con kernel en scikit-learn](#Análisis-de-componentes-principales-con-kernel-en-scikit-learn)
# - [Resumen](#Resumen)


# En[2]:


from IPython.display import Image


# # Reducción de dimensionalidad sin supervisión mediante el análisis de componentes principales

# ## Los pasos esenciales que se esconden detrás del análisis de componentes principales

# En[3]:




# ## Extraer los componentes principales paso a paso

# En[4]:


import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# si el conjunto de datos Wine está temporalmente fuera de servicio en el 
# repositorio de aprendizaje automático UCI, descomenta la siguiente línea 
# de código para cargar el conjunto de datos desde una ruta local:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()


# <hr>

# Separar los datos en subconjuntos 70% de entrenamiento y 30% de prueba.

# En[5]:


from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)


# Estandarizar los datos.

# En[6]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# ---
# 
# **Nota**
# 
# Po error, he escrito `X_test_std = sc.fit_transform(X_test)` en lugar de `X_test_std = sc.transform(X_test)`. En este caso, esto no daría una gran diferencia puesto que la media y la desviación estándar del conjunto de prueba debería ser (bastante) similar. Sin embargo, como recordarás del Capítulo 3, la manera correcta es reutilizar parámetros del conjunto de entrenamiento si estamos realizando algún tipo de transformación -- el conjunto de prueba debe representar básicamente datos "nuevos, no vistos".
# 
# Mi tipo de letra inicial refleja un error común y hay gente que *no* está reutilizando estos parámetros del modelo de entrenamiento/construcción y estandariza los nuevos datos "desde cero". Este es un sencillo ejemplo de por qué es un problema.
# 
# Supongamos que tenemos un simple conjunto de entrenamiento de 3 muestras con 1 característica (vamos a llamar a esta característica "length"):
# 
# - train_1: 10 cm -> class_2
# - train_2: 20 cm -> class_2
# - train_3: 30 cm -> class_1
# 
# mean: 20, std.: 8.2
# 
# Tras la estandarización, los valores de las características transformadas son
# 
# - train_std_1: -1.21 -> class_2
# - train_std_2: 0 -> class_2
# - train_std_3: 1.21 -> class_1
# 
# A continuación, supongamos que nuestro modelo ha aprendido a clasificar muestras con un valor de longitud estandarizada < 0.6 as class_2 (class_1 otherwise). Por el momento, todo bien. Ahora, digamos que tenemos 3 puntos de datos sin etiquetar que queremos clasificar:
# 
# - new_4: 5 cm -> class ?
# - new_5: 6 cm -> class ?
# - new_6: 7 cm -> class ?
# 
# Si miramos los valores de "length" sin estandarizar en nuestro conjunto de datos de entrenamiento, es lógico pensar que todas estas muestras probablemente pertenecen a class_2. Sin embargo, si los estandarizamos volviendo a calcular la desviación estándar y la media obtendrás valores similares a los de antes en el conjunto de entrenamiento y tu clasificador clasificará (probablemente de manera incorrecta) las muestras 4 y 5 como clase 2.
# 
# - new_std_4: -1.21 -> class 2
# - new_std_5: 0 -> class 2
# - new_std_6: 1.21 -> class 1
# 
# Sin embargo, si utilizamos los parámetros de tu "estandarización del conjunto de entrenamiento", obtendremos estos valores:
# 
# - sample5: -18.37 -> class 2
# - sample6: -17.15 -> class 2
# - sample7: -15.92 -> class 2
# 
# Los valores 5 cm, 6 cm y 7 cm son mucho más bajos que cualquier otro que hayamos visto previamente en el conjunto de entrenamiento. Así, lo único que tiene sentido es que las características estandarizadas de las "nuevas muestras" son mucho más bajas que cualquier característica estandarizada en el conjunto de entrenamiento.
# 
# ---

# Autodescomposición de la matriz de covarianza.

# En[7]:


import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)


# **Nota**: 
# 
# Anteriormente, he utilizado la función [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) para descomponer la matriz de covarianza simétrica en sus autovalores y autovectores.
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     No se trata de un "error", aunque probablemente no sea lo adecuado. En estos casos, sería mejor utilizar [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html), diseñada para [Matrices hermitianas](https://es.wikipedia.org/wiki/Matriz_hermitiana). Esta última siempre devuelve autovalores reales; a pesar de que la menos estable `np.linalg.eig` numéricamente hablando puede descomponer matrices cuadradas no simétricas, puedes ver que en determinados casos devuelve autovalores complejos. (S.R.)
# 


# ## Varianza total y explicada

# En[8]:


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# En[9]:


import matplotlib.pyplot as plt


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()



# ## Transformación de características

# En[10]:


# Hacer una lista de las tuplas (autovalor, autovector)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Ordenar las tuplas (autovalor, autovector) de mayor a menor
eigen_pairs.sort(key=lambda k: k[0], reverse=True)


# En[11]:


w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)


# **Nota**
# Según la versión de NumPy y LAPACK que estés utilizando, puedes obtener la Matriz W con sus signos cambiados. Ten en cuenta que esto no supone ningún problema: Si $v$ es un autovector de una matriz $\Sigma$, tenemos
# 
# $$\Sigma v = \lambda v,$$
# 
# donde $\lambda$ es nuestro autovalor,
# 
# 
# entonces $-v$ es también un autovector que tiene el mismo autovalor, donde
# $$\Sigma \cdot (-v) = -\Sigma v = -\lambda v = \lambda \cdot (-v).$$

# En[12]:


X_train_std[0].dot(w)


# En[13]:


X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()



# ## Análisis de componentes principales en scikit-learn

# **NOTA**
# 
# Los siguientes cuatro fragmentos de código han sido añadidos adicionalmente al contenido de este libro para mostrar cómo duplicar los resultados desde nuestra propia implementación PCA en scikit-learn:

# En[14]:


from sklearn.decomposition import PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# En[15]:


plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()


# En[16]:


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# En[17]:


plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


# En[18]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # Configurar generador de marcadores y mapa de colores
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Representar la superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # representar las muestras de clase
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


# Entrenar un clasificador de regresión logística con los dos dos primeros componentes.

# En[19]:


from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)


# En[20]:


plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()


# En[21]:


plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_05.png', dpi=300)
plt.show()


# En[22]:


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_



# # Compresión de datos supervisada mediante análisis discriminante lineal

# ## Análisis de componentes principales frente a análisis discriminante lineal

# En[4]:




# ## Cómo funciona interiormente el análisis discriminante lineal


# ## Calcular las matrices de dispersión

# Calcular los vectores medios para cada clase:

# En[24]:


np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))


# Calcular la matriz de dispersión dentro de las clases:

# En[25]:


d = 13 # número de características
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))


# Mejor: una matriz de covarianza puesto que las clases no están distribuidos equitativamente:

# En[26]:


print('Class label distribution: %s' 
      % np.bincount(y_train)[1:])


# En[27]:


d = 13  # número de características
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                     S_W.shape[1]))


# calcular la matriz de dispersión entre clases:

# En[28]:


mean_overall = np.mean(X_train_std, axis=0)
d = 13  # número de características
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))



# ## Seleccionar discriminantes lineales para el nuevo subespacio de características

# Resolver el problema del autovalor generalizado para la matriz $S_W^{-1}S_B$:

# En[29]:


eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# **Nota**:
#     
# Anteriormente, he utilizado la función [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) para descomponer la matriz de covarianza simétrica en sus autovalores y autovectores.
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     No se trata de un "error", aunque probablemente no sea lo adecuado. En estos casos, sería mejor utilizar [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html), diseñada para [Matrices hermitianas](https://es.wikipedia.org/wiki/Matriz_hermitiana). Esta última siempre devuelve autovalores reales; a pesar de que la menos estable `np.linalg.eig` numéricamente hablando puede descomponer matrices cuadradas no simétricas, puedes ver que en determinados casos devuelve autovalores complejos. (S.R.)
# 

# Ordenar los autovectores en orden decreciente de los autovalores:

# En[30]:


# Hacer una lista de tuplas (autovalor, autovector)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Ordenar las tuplas (autovalor, autovector) de mayor a menor
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Confirmar visualmente que la lista está ordenada de forma correcta mediante autovalores decrecientes

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# En[31]:


tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_07.png', dpi=300)
plt.show()


# En[32]:


w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)



# ## Proyectar muestras en el nuevo espacio de características

# En[33]:


X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('images/05_08.png', dpi=300)
plt.show()



# ## ADL con scikit-learn

# En[34]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)


# En[35]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_09.png', dpi=300)
plt.show()


# En[36]:


X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_10.png', dpi=300)
plt.show()



# # Utilizar el análisis de componentes principales con kernels para mapeos no lineales

# En[5]:





# ## Implementar un análisis de componentes principales con kernels en Python

# En[38]:


from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   

    """
    # Calcular pares de diatancias euclidianas al cuadrado
    # en el conjunto de datos dimensional MxN.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convertir los pares de distancias en una matriz cuadrada.
    mat_sq_dists = squareform(sq_dists)

    # Calcular la matriz de kernel simétrica.
    K = exp(-gamma * mat_sq_dists)

    # Centrar la matriz de kernel.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtener autopares a partir de la matriz de kernel centrada
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Recoger los primeros autovectores k (muestras proyectadas)
    X_pc = np.column_stack((eigvecs[:, i]
                            for i in range(n_components)))

    return X_pc



# ### Ejemplo 1: separar formas de media luna

# En[39]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('images/05_12.png', dpi=300)
plt.show()


# En[40]:


from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_13.png', dpi=300)
plt.show()


# En[41]:


X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_14.png', dpi=300)
plt.show()



# ### Ejemplo 2: separar círculos concéntricos

# En[42]:


from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('images/05_15.png', dpi=300)
plt.show()


# En[43]:


scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_16.png', dpi=300)
plt.show()


# En[44]:


X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_17.png', dpi=300)
plt.show()



# ## Proyectar nuevos puntos de datos

# En[45]:


from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   
     
     lambdas: list
       Eigenvalues

    """
    # Calcular pares de diatancias euclidianas al cuadrado
    # en el conjunto de datos dimensional MxN.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convertir los pares de distancias en una matriz cuadrada.
    mat_sq_dists = squareform(sq_dists)

    # Calcular la matriz de kernel simétrica.
    K = exp(-gamma * mat_sq_dists)

    # Centrar la matriz de kernel.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtener autopares a partir de la matriz de kernel centrada
    # scipy.linalg.eigh los devuelve en orden ascendente
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, i]
                              for i in range(n_components)))

    # Recoger los correspondientes autovalores
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas


# En[46]:


X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)


# En[47]:


x_new = X[25]
x_new


# En[48]:


x_proj = alphas[25] # original projection
x_proj


# En[49]:


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# proyección de una "nueva" muestra de datos 
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj 


# En[50]:


plt.scatter(alphas[y == 0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)

plt.tight_layout()
# plt.savefig('images/05_18.png', dpi=300)
plt.show()



# ## Análisis de componentes principales con kernel en scikit-learn

# En[51]:


from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
# plt.savefig('images/05_19.png', dpi=300)
plt.show()



# # Resumen

# ...
