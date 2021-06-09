
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 4 - Generar buenos modelos de entrenamiento: Preprocesamiento de datos

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Sumario

# - [Tratar con datos ausentes](#Tratar-con-datos-ausentes)
#   - [Identificar valores ausentes en datos tabulares](#Identificar-valores-ausentes-en-datos-tabulares)
#   - [Eliminar muestras o características con valores ausentes](#Eliminar-muestras-o-características-con-valores-ausentes)
#   - [Imputar valores ausentes](#Imputar-valores-ausentes)
#   - [Entender la API de estimador de scikit-learn](#Entender-la-API-de-estimador-de-scikit-learn)
# - [Trabajar con datos categóricos](#Trabajar-con-datos-categóricos)
#   - [Características nominales y ordinales](#Características-nominales-y-ordinales)
#   - [Mapear características ordinales](#Mapear-características-ordinales)
#   - [Codificar etiquetas de clase](#Codificar-etiquetas-de-clase)
#   - [Realizar una codificación en caliente sobre características nominales](#Realizar-una-codificación-en-caliente-sobre-características-nominales)
# - [Dividir un conjunto de datos en conjuntos de prueba y de entrenamiento individuales](#Dividir-un-conjunto-de-datos-en-conjuntos-de-prueba-y-de-entrenamiento-individuales)
# - [Ajustar las características a la misma escala](#Ajustar-las-características-a-la-misma-escala)
# - [Seleccionar características significativas](#Seleccionar-características-significativas)
#   - [Regularización L1 y L2 como penalizaciones contra la complejidad del modelo](#Regularización-L1-y-L2-como-penalizaciones-contra-la-complejidad-del-modelo)
#   - [Una interpretación geométrica de la regularización L2](#Una-interpretación-geométrica-de-la-regularización-L2)
#   - [Soluciones-dispersas-con-la-regularización-L1](#Soluciones-dispersas-con-la-regularización-L1)
#   - [Algoritmos de selección de características secuenciales](#Algoritmos-de-selección-de-características-secuenciales)
# - [Evaluar la importancia de las características con bosques aleatorios](#Evaluar-la-importancia-de-las-características-con-bosques-aleatorios)
# - [Resumen](#Resumen)


# In[2]:


from IPython.display import Image


# # Tratar con datos ausentes

# ## Identificar valores ausentes en datos tabulares

# En[3]:


import pandas as pd
from io import StringIO
import sys

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# Si estás utilizando Python 2.7, debes
# convertir la cadena a unicode:

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
df


# En[4]:


df.isnull().sum()


# En[5]:


# accede a la matriz subyacente NumPy
# mediante el atributo `values`
df.values



# ## Eliminar muestras o características con valores ausentes

# En[6]:


# elimina filas que contienen valores ausentes

df.dropna(axis=0)


# En[7]:


# elimina columnas que contienen valores ausentes

df.dropna(axis=1)


# In[8]:


# elimina columnas que contienen valores ausentes

df.dropna(axis=1)


# En[9]:


# solo se descartan filas donde todas las columnas son NaN

df.dropna(how='all')  


# En[10]:


# se descartan filas que tienen menos de 4 valores reales 

df.dropna(thresh=4)


# En[11]:


# solo se descartan filas donde NaN aparece en determinadas columnas (aquí: 'C')

df.dropna(subset=['C'])



# ## Imputar valores ausentes

# En[12]:


# de nuevo: nuestra matriz original
df.values


# En[13]:


# imputa valores ausentes mediante el valor medio de la columna

from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data



# ## Entender la API de estimador de scikit-learn

# En[14]:




# En[15]:





# # Trabajar con datos categóricos

# ## Características nominales y ordinales

# En[16]:


import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
df



# ## Mapear características ordinales

# En[17]:


size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df


# En[18]:


inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)



# ## Codificar etiquetas de clase

# En[19]:


import numpy as np

# crea un diccionario de mapeo 
# para convertir etiquetas de clase de cadenas a enteros
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping


# En[20]:


# para convertir etiquetas de clase de cadenas a enteros
df['classlabel'] = df['classlabel'].map(class_mapping)
df


# En[21]:


# revierte el mapeo de las etiquetas de clase 
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df


# En[22]:


from sklearn.preprocessing import LabelEncoder

# Codificación de etiquetas con LabelEncoder de sklearn
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y


# En[23]:


# revierte el mapeo
class_le.inverse_transform(y)



# ## Realizar una codificación en caliente sobre características nominales

# En[24]:


X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X


# En[25]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()


# En[26]:


# devuelve una matriz regular para que podamos omitir 
# el paso toarray 

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
ohe.fit_transform(X)


# En[27]:


# codificación en caliente con pandas

pd.get_dummies(df[['price', 'color', 'size']])


# En[28]:


# multicolinearidad protegida en get_dummies

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)


# En[29]:


# multicolinearidad protegida para el OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()[:, 1:]



# # Dividir un conjunto de datos en conjuntos de prueba y de entrenamiento individuales

# En[30]:


df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# si el conjunto de datos Wine está temporalmente fuera de servicio en el 
# repositorio de aprendizaje automático UCI, descomenta la siguiente línea 
# de código para cargar el conjunto de datos desde una ruta local:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()


# En[31]:


from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # Ajustar las características a la misma escala

# En[32]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# En[33]:


from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# Un ejemplo visual:

# En[34]:


ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

# Ten en cuenta que pandas utiliza ddof=1 (desviación estándar de la muestra) 
# por defecto, mientras que el método std y StandardScaler de NumPy
# utiliza ddof=0 (desviación estándar de la población)

# normaliza
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))



# # Seleccionar características significativas

# ...

# ## Regularización L1 y L2 como penalizaciones contra la complejidad del modelo

# ## Una interpretación geométrica de la regularización L2

# En[35]:




# En[36]:




# ## Soluciones dispersas con la regularización L1

# En[37]:




# Para los modelos redularizados que soportan la regularización L1, simplemente debemos ajustar el parámetro `penalty` a `'l1'` para obtener una solución dispersa:

# En[38]:


from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')


# Aplicado a los datos Wine estandarizados...

# En[39]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# En[40]:


lr.intercept_


# En[41]:


np.set_printoptions(8)


# En[42]:


lr.coef_[lr.coef_!=0].shape


# En[43]:


lr.coef_


# En[44]:


import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
plt.show()



# ## Algoritmos de selección de características secuenciales

# En[45]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# En[46]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# seleccionar características 
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# representar el rendimiento de subconjuntos de características 
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()


# En[47]:


k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])


# En[48]:


knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))


# En[49]:


knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))



# # Evaluar la importancia de las características con bosques aleatorios

# En[50]:


from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()


# En[51]:


from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of samples that meet this criterion:', 
      X_selected.shape[0])


# Ahora, vamos a imprimir las 3 características que cumplen con el criterio de umbral para selección de características que hemos definido anteriormente (observa que este fragmento de código no aparece en el libro, sino que ha sido añadido posteriormente con propósitos ilustrativos):

# En[52]:


for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))



# # Resumen

# ...
