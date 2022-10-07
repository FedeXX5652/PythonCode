
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 8 - Aplicar el aprendizaje automático para el análisis de sentimiento

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Sumario

# - [Preparar los datos de críticas de cine de IMDb para el procesamiento de texto](#Preparar-los-datos-de-críticas-de-cine-de-IMDb-para-el-procesamiento-de-texto)
#   - [Obtener el conjunto de datos de críticas de cine](#Obtener-el-conjunto-de-datos-de-críticas-de-cine)
#   - [Preprocesar el conjunto de datos de películas en un formato adecuado](#Preprocesar-el-conjunto-de-datos-de-películas-en-un-formato-adecuado)
# - [Introducir el modelo "bolsa de palabras"](#Introducir-el-modelo-"bolsa-de-palabras")
#   - [Transformar palabras en vectores de características](#Transformar-palabras-en-vectores-de-características)
#   - [Evaluar la relevancia de las palabras mediante frecuencia de término – frecuencia inversa de documento](#Evaluar-la-relevancia-de-las-palabras-mediante-frecuencia-de-término-frecuencia-inversa-de-documento)
#   - [Limpiar datos textuales](#Limpiar-datos-textuales)
#   - [Procesar documentos en componentes léxicos](#Procesar-documentos-en-componentes-léxicos)
# - [Entrenar un modelo de regresión logística para clasificación de documentos](#Entrenar-un-modelo-de-regresión-logística-para-clasificación-de-documentos)
# - [Trabajar con datos más grandes: algoritmos online y aprendizaje out-of-core](#Trabajar-con-datos-más-grandes-algoritmos-online-y-aprendizaje-out-of-core)
# - [Modelado de temas con Latent Dirichlet Allocation](#Modelado-de-temas-con-Latent-Dirichlet-Allocation)
#   - [Descomponer documentos de textos con LDA](#Descomponer-documentos-de-textos-con-LDA)
#   - [LDA con scikit-learn](#LDA-con-scikit-learn)
# - [Resumen](#Resumen)


# # Preparar los datos de críticas de cine de IMDb para el procesamiento de texto 

# ## Obtener el conjunto de datos de críticas de cine

# Puedes descargar un archivo comprimido del conjunto de datos de críticas de cine desde [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
# Una vez descargado el conjunto de datos, debes descomprimir los archivos.
# 
# A) Si trabajas con Linux o macOS, abre una nueva ventana de terminal, escribe cd en el directorio de descarga y ejecuta 
# 
# `tar -zxf aclImdb_v1.tar.gz`
# 
# B) Si trabajas con Windows, puedes descargar un compactador de archivos gratuito como [7Zip](http://www.7-zip.org) para extraer los archivos después de la descarga.

# **Código opcional para descargar y descomprimir el conjunto de datos con Python:**

# En[2]:


import os
import sys
import tarfile
import time


source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size
    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()


if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
    
    if (sys.version_info < (3, 0)):
        import urllib
        urllib.urlretrieve(source, target, reporthook)
    
    else:
        import urllib.request
        urllib.request.urlretrieve(source, target, reporthook)


# En[3]:


if not os.path.isdir('aclImdb'):

    with tarfile.open(target, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)


# ## Preprocesar el conjunto de datos de películas en un formato adecuado

# En[23]:


import pyprind
import pandas as pd
import os

# cambiar el `basepath` al directorio del
# conjunto de datos de películas descomprimido

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


# Barajar el DataFrame:

# En[24]:


import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))


# Opcional: guardar los datos juntados como un archivo CSV:

# En[25]:


df.to_csv('movie_data.csv', index=False, encoding='utf-8')


# En[26]:


import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)


# <hr>
# ### Nota
# 
# Si has tenido algún problema al crear el archivo `movie_data.csv` en el capítulo anterior, encontrarás un archivo comprimido para descargar en 
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/tree/master/code/ch08/
# <hr>


# # Introducir el modelo "bolsa de palabras"

# ...

# ## Transformar palabras en vectores de características

# Mediante la llamada del método fit_transform en CountVectorizer, hemos construido simplemente el vocabulario del modelo de "bolsa de palabras" y hemos transformado las siguientes tres frases en vectores de características dispersos:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
# 

# En[6]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)


# Ahora vamos a imprimir el contenido del vocabulario para comprender mejor los conceptos subyacentes:

# En[7]:


print(count.vocabulary_)


# Como podemos ver tras ejecutar el comando anterior, el vocabulario está almacenado en un diccionario de Python que mapea las palabras únicas en índice enteros. A continuación, vamos a imprimir los vectores de características que acabamos de crear:

# Cada posición indexada en los vectores de características que aparece corresponde a los valores enteros almacenados como elementos de diccionario en el vocabulario CountVectorizer. Por ejemplo, la primera característica en la posición indexada 0 se parece al recuento de la palabra 'and', que solo aparece en el último documento, y la palabra 'is', en la posición indexada 1 (la segunda característica en los vectores de documento), aparece en las tres frases. Estos valores en los vectores de características también se conocen como frecuencias de término sin procesar *tf (t,d)*—el número de veces que un término t aparece en un documento *d*.

# En[8]:


print(bag.toarray())



# ## Evaluar la relevancia de las palabras mediante frecuencia de término – frecuencia inversa de documento

# En[9]:


np.set_printoptions(precision=2)


# Cuando analizamos datos textuales, a menudo encontramos palabras que aparecen en múltiples documentos de amblas clases. Normalmente, estas palabras no contienen información útil o discriminatoria. En esta subsección, conoceremos una técnica útil denominada frecuencia de término – frecuencia inversa de documento (tf-idf, del inglés term frequency-inverse document frequency) que se puede utilizar para bajar de peso estas palabras que aparecen con frecuencia en los vectores de características. El tf-idf se puede definir como el producto de la frecuencia de término y la frecuencia inversa de documento:
# 
# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$
# 
# Aquí tf(t, d) es la frecuencia de término que presentamos en la sección anterior,
# y la frecuencia inversa de documento *idf(t, d)* puede calcularse del siguiente modo:
# 
# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$
# 
# donde $n_d$ es el número total de documentos y *df(d, t)* es el número de documentos *d* que contienen el término *t*. Ten en cuenta que añadir la constante 1 al denominador es opcional y sirve para asignar un valor diferente a cero a los términos que aparecen en todas las muestras de entrenamiento; el logaritmo se utiliza para garantizar que las bajas frecuencias de documento no adquieran demasiado peso.
# 
# Scikit-learn ya implementa otro transformador, la clase `TfidfTransformer`, que toma las frecuencias de término sin procesar de la clase `CountVectorizer` como entrada y las transforma en tf-idf::

# En[10]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())


# Como hemos visto en la subsección anterior, la palabra 'is' tiene la frecuencia de término más grande en el tercer documento, siendo así la palabra que aparece con mayor frecuencia. Sin embargo, después de transformar el mismo vector de características en tf-idf,vemos que la palabra 'is' está
# está asociado con un tf-idf relativamente pequeño (0.45) en el tercer documento, ya que también está
# presente en el primer y segundo documento y, por lo tanto, es poco probable que contenga información discriminatoria útil.
# 

# Sin embargo, si hubiéramos calculado manualmente los tf-idf de los términos individuales en nuestros vectores de características, habríamos visto que TfidfTransformer calcula los tf-idf de manera ligeramente distinta a las ecuaciones de manual estándar que hemos definido anteriormente. Las ecuaciones para la frecuencia inversa de documento implementadas en scikit-learn se calculan de la siguiente forma:

# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
# 
# La ecuación tf-idf que hemos implementado en scikit-learn es así:
# 
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
# 
# Si bien también es habitual normalizar las frecuencias de término sin procesar antes de calcular los tf-idfs, la clase TfidfTransformer normaliza los tf-idf directamente.
# 
# Por defecto (norm='l2'), el TfidfTransformer de scikit-learn aplica la normalización L2, que devuelve un vector de longitud 1 dividiendo un vector de característica no normalizado v por su normativa L2:
# 
# $$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
# 
# Para asegurarnos de que hemos entendido cómo trabaja el TfidfTransformer, veamos
# un ejemplo en el cual calculamos el tf-idf de la palabra 'is' en el tercer documento.
# 
# La palabra 'is' tiene una frecuencia de término de 3 (tf=3) en el tercer documento, y la frecuencia de documento de este término es 3 puesto que el término 'is' aparece en los tres documentos (df=3). Por tanto, podemos calcular la frecuencia inversa de documento del siguiente modo:
# 
# $$\text{idf}("is", d3) = log \frac{1+3}{1+3} = 0$$
# 
# A continuación, para calcular el tf-idf, simplemente debemos añadir 1 a la frecuencia inversa de documento y multiplicarla por la frecuencia de término:
# 
# $$\text{tf-idf}("is",d3)= 3 \times (0+1) = 3$$

# En[11]:


tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)


# Si repetimos esta operación para todos los términos en el tercer documento, obtendremos los siguientes vectores tf-idf: [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0, 1.69, 1.29]. Sin embargo, ten en cuenta que los valores en este vector de características son diferentes a los valores que hemos obtenido del TfidfTransformer que hemos utilizado anteriormente. El paso final que nos falta en este cálculo del tf-idf es la normalización L2, que podemos aplicar como sigue:

# $$\text{tfi-df}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2, 3.0^2, 3.39^2, 1.29^2, 1.29^2, 1.29^2, 2.0^2 , 1.69^2, 1.29^2]}}$$
# 
# $$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$
# 
# $$\Rightarrow \text{tfi-df}_{norm}("is", d3) = 0.45$$

# Como podemos ver, los resultados coinciden con los resultados devueltos por el TfidfTransformer de scikit-learn y, como ya sabemos cómo se calculan los tf-idf, pasaremos a la siguiente sección y aplicaremos dichos conceptos al conjunto de datos de críticas de cine.

# En[12]:


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf 


# En[13]:


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf



# ## Limpiar datos textuales

# En[14]:


df.loc[0, 'review'][-50:]


# En[15]:


import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# En[16]:


preprocessor(df.loc[0, 'review'][-50:])


# In[17]:


preprocessor("</a>This :) is :( a test :-)!")


# En[18]:


df['review'] = df['review'].apply(preprocessor)



# ## Procesar documentos en componentes léxicos

# En[9]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# En[10]:


tokenizer('runners like running and thus they run')


# En[11]:


tokenizer_porter('runners like running and thus they run')


# En[12]:


import nltk

nltk.download('stopwords')


# En[13]:


from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]



# # Entrenar un modelo de regresión logística para clasificación de documentos

# Separa HTML y puntuación para acelerar el GridSearch más adelante:

# En[24]:


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# En[25]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


# **Nota importante acerca de `n_jobs`**
# 
# Ten en cuenta que es muy recomendable utilizar `n_jobs=-1` (en lugar de `n_jobs=1`) en el código de ejemplo anterior para utilizar todos los núcleos disponibles en tu ordenador y acelerar la búsqueda de cuadrículas. Sin embargo, hay usuarios de Windows que han informado de errores al ejecutar el código anterior con la configuración n_jobs=-1 con relación al decapado de las funciones tokenizer y tokenizer_porter para el multiprocesado en Windows. Otra solución alternativa sería reemplazar esas dos funciones, `[tokenizer, tokenizer_porter]`, por `[str.split]`. Sin embargo, observa que el reemplazo por el simple str.split no soportaría las declinaciones.

# **Nota importante acerca del tiempo de ejecución**
# 
# La ejecución del siguiente fragmento de código **puede tardar entre 30 y 60 min** según el ordenador, pues según el parámetro de cuacdrículo que hemos definido hay 2*2*2*3*5 + 2*2*2*3*5 = 240 modelos para definir.
# 
# Si no quieres esperar tanto, puedes disminuir el tamaño del conjunto de datos reduciendo el número de muestras de entrenamiento, por ejemplo, del modo siguiente:
# 
#     X_train = df.loc[:2500, 'review'].values
#     y_train = df.loc[:2500, 'sentiment'].values
#     
# Sin embargo, ten en cuenta que reducir el tamaño del conjunto de entrenamiento a un valor probablemente dará como resultado modelos con un rendimiento más bajo. De forma alternativa, puedes eliminar parámetros de la cuadrícula anterior para reducir el número de modelos a definir, por ejemplo, utilizando lo siguiente:
# 
#     param_grid = [{'vect__ngram_range': [(1, 1)],
#                    'vect__stop_words': [stop, None],
#                    'vect__tokenizer': [tokenizer],
#                    'clf__penalty': ['l1', 'l2'],
#                    'clf__C': [1.0, 10.0]},
#                   ]

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
    gs_lr_tfidf.verbose=2
    X_train = df.loc[:250, 'review'].values
    y_train = df.loc[:250, 'sentiment'].values
    X_test = df.loc[25000:25250, 'review'].values
    y_test = df.loc[25000:25250, 'sentiment'].values


# En[26]:


gs_lr_tfidf.fit(X_train, y_train)


# En[27]:


print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)


# En[28]:


clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# <hr>
# <hr>

# ####  Inicio del comentario:
#     
# Observa que `gs_lr_tfidf.best_score_` es la puntuación media de la validación cruzada de k iteracioness. Por ejemplo, si tenemos un objeto `GridSearchCV` con una validación cruzada de 5 iteraciones (como la anterior), el atributo `best_score_` devuelve la puntuación media sobre las 5 iteraciones del mejor modelo. Podemos ilustrar esto con un ejemplo:

# En[29]:


from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)

cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=0).split(X, y))
    
cross_val_score(LogisticRegression(random_state=123), X, y, cv=cv5_idx)


# Ejecutando el código anterior, hemos creado un conjunto de datos simple de enteros aleatorios que representarán nuestras etiquetas de clase. A continuación, hemos alimentado los índices de 5 iteraciones de validación cruzada (`cv3_idx`) para el puntuador  `cross_val_score`, que ha devuelto 5 puntuaciones de precisión: hay 5 valores de precisión para las 5 iteraciones de prueba.  
# 
# Seguidamente, vamos a utilizar el objeto `GridSearchCV` y a proporcionarle los mismos conjuntos de 5 validaciones cruzadas (mediante los índices pregenerados `cv3_idx`):

# En[30]:


from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(LogisticRegression(), {}, cv=cv5_idx, verbose=3).fit(X, y) 


# Como podemos ver, las puntuaciones para las 5 iteraciones son exactamente las mismas que las obtenidas de `cross_val_score` anteriormente.

# Ahora, el atributo best_score_ attribute del objeto `GridSearchCV`, que pasa a estar disponible después de `fit`ting, devuelve la puntuación de precisión media  del mejor modelo:

# En[31]:


gs.best_score_


# Como podemos ver, el resultado anterior es coherente con la puntuación media calculada por el `cross_val_score`.

# En[32]:


cross_val_score(LogisticRegression(), X, y, cv=cv5_idx).mean()


# #### Final del comentario.
# 
# <hr>
# <hr>


# # Trabajar con datos más grandes: algoritmos online y aprendizaje out-of-core

# En[27]:


import numpy as np
import re
from nltk.corpus import stopwords

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# En[28]:


next(stream_docs(path='movie_data.csv'))


# En[29]:


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# En[30]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='movie_data.csv')


# **Nota**
# 
# - Puedes sustituir `Perceptron(n_iter, ...)` por `Perceptron(max_iter, ...)` en scikit-learn >= 0.19. El parámetro `n_iter` se utiliza aquí deliberadamente, porque scikit-learn 0.18 todavía se utiliza mucho.
# 

# En[31]:


import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# En[32]:


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))


# En[33]:


clf = clf.partial_fit(X_test, y_test)


# ## Modelado de temas con Latent Dirichlet Allocation

# ### Descomponer documentos de textos con LDA

# ### LDA con scikit-learn

# En[1]:


import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)


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
    df.loc[:500].to_csv('movie_data.csv')
    df = pd.read_csv('movie_data.csv', nrows=500)
    print('SMALL DATA SUBSET CREATED FOR TESTING')


# En[2]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)


# En[3]:


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)


# En[4]:


lda.components_.shape


# En[5]:


n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))


# Basándonos en la lectura de las 5 palabras más importantes de cada tema, podemos adivinar que el LDA ha identificado los siguientes temas:
#     
# 1. Películas generalmente malas (realmente no es una categoría de tema)
# 2. Películas sobre familias
# 3. Películas bélicas
# 4. Películas de autor
# 5. Películas policíacas
# 6. Películas de terror
# 7. Películas de comedia
# 8. Películas de alguna manera relacionadas con programas de telev
# 9. Películas basadas en libros
# 10. Películas de acción

# Para confirmar que las categorías tienen sentido basadas en las críticas, vamos a representar tres películas de la categoría películas de terror (la películas de terror pertenecen a la categoría 6, en la posición de índice 5):

# En[6]:


horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')


# Con el ejemplo de código anterior, hemos mostrado los primeros 300 caracteres de las primeras tres películas de terror, y podemos ver que las críticas —incluso si no sabemos a qué película exactamente pertenecen— parecen críticas de películas de terror (sin embargo, alguien podría argumentar que Horror movie #2 también podría ser un buen ajuste para la categoría de tema 1: Generally bad movies).


# # Resumen
