
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 6 - Aprender las buenas prácticas para la evaluación de modelos y el ajuste de hiperparámetros

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# En[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Sumario

# - [Simplificar flujos de trabajo con pipelines](#Simplificar-flujos-de-trabajo-con-pipelines)
#   - [Cargar el conjunto de datos Breast Cancer Wisconsin](#Cargar-el-conjunto-de-datos-Breast-Cancer-Wisconsin)
#   - [Combinar transformadores y estimadores en un pipeline](#Combinar-transformadores-y-estimadores-en-un-pipeline)
# - [Utilizar la validación cruzada de K iteraciones para evaluar el rendimiento de un modelo](#Utilizar-la-validación-cruzada-de-K-iteraciones-para-evaluar-el-rendimiento-de-un-modelo)
#   - [El método de retención](#El-método-de-retención)
#   - [Validación cruzada de k iteraciones](#Validación-cruzada-de-k-iteraciones)
# - [Depurar algoritmos con curvas de validación y aprendizaje](#Depurar-algoritmos-con-curvas-de-validación-y-aprendizaje)
#   - [Diagnosticar problemas de sesgo y varianza con curvas de aprendizaje](#Diagnosticar-problemas-de-sesgo-y-varianza-con-curvas-de-aprendizaje)
#   - [Resolver el sobreajuste y el subajuste con curvas de validación](#Resolver-el-sobreajuste-y-el-subajuste-con-curvas-de-validación)
# - [Ajustar los modelos de aprendizaje automático con la búsqueda de cuadrículas](#Ajustar-los-modelos-de-aprendizaje-automático-con-la-búsqueda-de-cuadrículas)
#   - [Ajustar hiperparámetros con la búsqueda de cuadrículas](#Ajustar-hiperparámetros-con-la-búsqueda-de-cuadrículas)
#   - [Selección de algoritmos con validación cruzada anidada](#Selección-de-algoritmos-con-validación-cruzada-anidada)
# - [Observar diferentes métricas de evaluación de rendimiento](#Observar-diferentes-métricas-de-evaluación-de-rendimiento)
#   - [Leer una matriz de confusión](#Leer-una-matriz-de-confusión)
#   - [Optimizar la exactitud y la exhaustividad de un modelo de clasificación](#Optimizar-la-exactitud-y-la-exhaustividad-de-un-modelo-de-clasificación)
#   - [Representar una característica operativa del receptor](#Representar-una-característica-operativa-del-receptor)
#   - [Métricas de calificación para clasificaciones multiclase](#Métricas-de-calificación-para-clasificaciones-multiclase)
# - [Tratar con el desequilibrio de clases](#Tratar-con-el-desequilibrio-de-clases)
# - [Resumen](#Resumen)


# En[2]:


from IPython.display import Image


# # Simplificar flujos de trabajo con pipelines

# ...

# ## Cargar el conjunto de datos Breast Cancer Wisconsin

# En[3]:


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

# si el conjunto de datos Breast Cancer Wisconsin está temporalmente fuera de servicio en el 
# repositorio de aprendizaje automático UCI, descomenta la siguiente línea 
# de código para cargar el conjunto de datos desde una ruta local:

# df_wine = pd.read_csv('wdbc.data', header=None)

df.head()


# En[4]:


df.shape


# <hr>

# En[5]:


from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_


# En[6]:


le.transform(['M', 'B'])


# En[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)



# ## Combinar transformadores y estimadores en un pipeline

# En[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


# En[9]:





# # Utilizar la validación cruzada de K iteraciones para evaluar el rendimiento de un modelo

# ...

# ## El método de retención

# En[10]:





# ## Validación cruzada de k iteraciones

# En[11]:




# En[12]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
    

kfold = StratifiedKFold(n_splits=10,
                        random_state=1).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# En[13]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



# # Depurar algoritmos con curvas de validación y aprendizaje


# ## Diagnosticar problemas de sesgo y varianza con curvas de aprendizaje

# En[14]:




# En[15]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1))

train_sizes, train_scores, test_scores =                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi=300)
plt.show()



# ## Resolver el sobreajuste y el subajuste con curvas de validación

# En[16]:


from sklearn.model_selection import validation_curve


param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()



# # Ajustar los modelos de aprendizaje automático con la búsqueda de cuadrículas


# ## Ajustar hiperparámetros con la búsqueda de cuadrículas 

# En[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# En[18]:


clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))



# ## Selección de algoritmos con validación cruzada anidada

# En[19]:




# En[20]:


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))


# En[21]:


from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), 
                                      np.std(scores)))



# # Observar diferentes métricas de evaluación de rendimiento

# ...

# ## Leer una matriz de confusión

# En[22]:




# En[23]:


from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# En[24]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()


# ### Nota adicional

# Recuerda que anteriormente hemos codificado las etiquetas de clase de manera que las muestras *malignas* tengan la clase "positivo"(1), y que las muestras *benignas* tengan la clase "negativo" (0):

# En[25]:


le.transform(['M', 'B'])


# En[26]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# A continuación, hemos impreso la matriz de confusión de este modo:

# En[27]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# Observa que las muestras 0 de clase (verdadero) que han sido predichas correctamente como clase 0 (verdaderos negativo) ahora están en la esquina superior izquierda de la matriz (índice 0, 0). Para cambiar el orden de manera que los verdaderos negativos estén en la esquina inferior derecha (índice 1,1) y los verdaderos positivos estén en la esquina superior izquierda, podemos utilizar el argumento `labels` como se muestra a continuación:

# En[28]:


confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
print(confmat)


# Para terminar:
# 
# Suponiendo que la clase 1 (maligno) es la clase positiva en este ejemplo, nuestro modelo clasificado correctamente como 71 de las muestras que pertenece a la clase 0 (verdadero negativo) y las 40 muestras que pertenecen a la clase 1 (verdadero positivo), respectivamente. Sin embargo, nuestro modelo también ha clasificado erróneamemte 1 muestra de la clase 0 como clase 1 (falso positivo), y esto ha predicho que 2 muestras son benignas aunque se trate de un tumor maligno (falso negativo).


# ## Optimizar la exactitud y la exhaustividad de un modelo de clasificación

# En[29]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


# En[30]:


from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)



# ## Representar una característica operativa del receptor

# En[31]:


from sklearn.metrics import roc_curve, auc
from scipy import interp

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2', 
                                           random_state=1, 
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]
    

cv = list(StratifiedKFold(n_splits=3, 
                          random_state=1).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('images/06_10.png', dpi=300)
plt.show()



# ## Métricas de calificación para clasificaciones multiclase

# En[32]:


pre_scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')


# ## Tratar con el desequilibrio de clases

# En[33]:


X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))


# En[34]:


y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100


# En[35]:


from sklearn.utils import resample

print('Number of class 1 samples before:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

print('Number of class 1 samples after:', X_upsampled.shape[0])


# In[36]:


X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))


# In[37]:


y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100



# # Resumen

# ...

# ---
# 
# El lector debe ignorar la siguiente celda.

# En[ ]:




# En[ ]:




