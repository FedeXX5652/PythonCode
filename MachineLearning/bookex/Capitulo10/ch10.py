
# coding: utf-8

# *Aprendizaje automático con Python 2ª edición* de [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repositorio de código: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencia de código: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Aprendizaje automático con Python - Códigos de ejemplo

# # Capítulo 10 - Predicción de variables de destino continuas con análisis de regresión

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[1]:




# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# El paquete seaborn, una librería de visualización creada en la parte superior de matplotlib, se puede instalar mediante
# 
#     conda install seaborn
# 
# o 
# 
#     pip install seaborn


# ### Sumario

# - [Introducción a la regresión lineal](#Introducción-a-la-regresión-lineal)
#   - [Regresión lineal simple](#Regresión-lineal-simple)
# - [Explorar el conjunto de datos Housing](#Explorar-el-conjunto-de-datos-Housing)
#   - [Cargar el conjunto Housing en un marco de datos](#Cargar-el-conjunto-Housing-en-un-marco-de-datos)
#   - [Visualizar las características importantes de un conjunto de datos](#Visualizar-las-características-importantes-de-un-conjunto-de-datos)
# - [Implementar un modelo de regresión lineal de mínimos cuadrados ordinarios](#Implementar-un-modelo-de-regresión-lineal-de-mínimos-cuadrados-ordinarios)
#   - [Resolver la regresión para parámetros de regresión con el descenso del gradiente](#Resolver-la-regresión-para-parámetros-de-regresión-con-el-descenso-del-gradiente)
#   - [Estimar el coeficiente de un modelo de regresión con scikit-learn](#Estimar-el-coeficiente-de-un-modelo-de-regresión-con-scikit-learn)
# - [Ajustar un modelo de regresión robusto con RANSAC](#Ajustar-un-modelo-de-regresión-robusto-con-RANSAC)
# - [Evaluar el rendimiento de los modelos de regresión lineal](#Evaluar-el-rendimiento-de-los-modelos-de-regresión-lineal)
# - [Utilizar métodos regularizados para regresión](#Utilizar-métodos-regularizados-para-regresión)
# - [Convertir un modelo de regresión lineal en una curva: la regresión polinomial](#Convertir-un-modelo-de-regresión-lineal-en-una-curva---la-regresión-polinomial)
#   - [Modelar relaciones no lineales en el conjunto de datos Housing](#Modelar-relaciones-no-lineales-en-el-conjunto-de-datos-Housing)
#   - [Tratar con relaciones no lineales mediante bosques aleatorios](#Tratar-con-relaciones-no-lineales-mediante-bosques-aleatorios)
#     - [Regresión de árbol de decisión](#Regresión-de-árbol-de-decisión)
#     - [Regresión con bosques aleatorios](#Regresión-con-bosques-aleatorios)
# - [Resumen](#Resumen)


# En[2]:


from IPython.display import Image


# # Introducción a la regresión lineal

# ## Regresión lineal simple

# En[3]:




# ## Regresión lineal múltiple

# En[4]:





# # Explorar el conjunto de datos Housing

# ## Cargar el conjunto Housing en un marco de datos

# Descripción, previamente disponible en: [https://archive.ics.uci.edu/ml/datasets/Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
# 
# Atributos:
#     
# <pre>
# CRIM: Tasa de criminalidad per cápita por ciudad
# ZN: Proporción de suelo residencial ocupado para terrenos de más de 2.300 m2
# INDUS: Proporción de hectáreas de negocios no minoristas por ciudad
# CHAS: Variable ficticia de Charles River (= 1 si el tramo limita el río; si no =0)
# NOX: Concentración de óxido nítrico (partes por 10 millones)
# RM: Número medio de habitaciones por vivienda
# AGE: Proporción de unidades ocupadas por sus propietarios construidas antes
	   de 1940
# DIS: Distancias ponderadas hasta cinco centros de empleo de Boston
# RAD: Índice de accesibilidad a autopistas radiales
# TAX: Tasa de impuesto a la propiedad de valor total de 10.000 $
# PTRATIO: Ratio de profesor por alumno por ciudad
# B: 1000(Bk - 0.63)^2, donde Bk es la proporción de [personas de ascendencia
  afroamericana] por ciudad
# LSTAT: Porcentaje de estatus inferior de la población
# MEDV: Valor medio de casas ocupadas por sus propietarios en 1000 $
# </pre>

# En[5]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# <hr>
# 
# ### Nota:
# 
# 
# Puedes encontrar una copia del conjunto de datos Housing (así como del resto de conjuntos de datos que se utilizan en este libro) con el código de acceso que encontrarás en la primera página del libro, el cual puedes utilizar si estás trabajando offline o si el conjunto de datos ubicado en https://raw.githubusercontent.com/rasbt/pythonmachine- learning-book-2nd-edition/master/code/ch10/ housing.data.txt está temporalmente fuera de servicio en el servidor de la UCI. Por ejemplo, para cargar el conjunto de datos Housing desde un directorio local, puedes sustituir estas líneas
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#                  'machine-learning-databases'
#                  '/housing/housing.data',
#                  sep='\s+')
# en el siguiente código de ejemplo por 
# df = pd.read_csv('./housing.data',
#                  sep='\s+')


# ## Visualizar las características importantes de un conjunto de datos

# En[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# En[7]:


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()


# En[8]:


import numpy as np


cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('images/10_04.png', dpi=300)
plt.show()



# # Implementar un modelo de regresión lineal de mínimos cuadrados ordinarios

# ...

# ## Resolver la regresión para parámetros de regresión con el descenso del gradiente

# En[9]:


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# En[10]:


X = df[['RM']].values
y = df['MEDV'].values


# En[11]:


from sklearn.preprocessing import StandardScaler


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# En[12]:


lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# En[13]:


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.tight_layout()
#plt.savefig('images/10_05.png', dpi=300)
plt.show()


# En[14]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 


# En[15]:


lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

plt.savefig('images/10_06.png', dpi=300)
plt.show()


# En[16]:


print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# En[17]:


num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))



# ## Estimar el coeficiente de un modelo de regresión con scikit-learn

# En[18]:


from sklearn.linear_model import LinearRegression


# En[19]:


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# En[20]:


lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

#plt.savefig('images/10_07.png', dpi=300)
plt.show()


# Alternativa de **Ecuaciones lineales**:

# en[21]:


# añadir un vector de columnas de "unos"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])



# # Ajustar un modelo de regresión robusto con RANSAC

# En[22]:


from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

#plt.savefig('images/10_08.png', dpi=300)
plt.show()


# En[23]:


print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)



# # Evaluar el rendimiento de los modelos de regresión lineal

# En[24]:


from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# En[25]:


slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# En[26]:


import numpy as np
import scipy as sp

ary = np.array(range(100000))


# En[27]:




# En[28]:




# En[29]:




# En[30]:


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/10_09.png', dpi=300)
plt.show()


# En[31]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



# # Utilizar métodos regularizados para regresión

# En[32]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(lasso.coef_)


# En[33]:


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# Regresión Ridge:

# En[34]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)


# Regresión LASSO:

# En[35]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)


# Regresión Elastic Net:

# En[36]:


from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)



# # Convertir un modelo de regresión lineal en una curva: la regresión polinomial

# En[37]:


X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])


# En[38]:


from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)


# En[39]:


# ajustar características lineales
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# ajustar características cuadráticas
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# representar resultados 
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/10_10.png', dpi=300)
plt.show()


# En[40]:


y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)


# En[41]:


print('Training MSE linear: %.3f, quadratic: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))



# ## Modelar relaciones no lineales en el conjunto de datos Housing

# En[42]:


X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# crear características cuadráticas
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# ajustar características 
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# representar resultados 
plt.scatter(X, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

#plt.savefig('images/10_11.png', dpi=300)
plt.show()


# Transformar el conjunto de datos:

# En[43]:


X = df[['LSTAT']].values
y = df['MEDV'].values

# transformar características
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# ajustar características
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# representar resultados
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')

plt.tight_layout()
#plt.savefig('images/10_12.png', dpi=300)
plt.show()



# # Tratar con relaciones no lineales mediante bosques aleatorios

# ...

# ## Regresión de árbol de decisión

# En[44]:


from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
#plt.savefig('images/10_13.png', dpi=300)
plt.show()



# ## Regresión con bosques aleatorios

# En[45]:


X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)


# En[46]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# En[47]:


plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='training data')
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/10_14.png', dpi=300)
plt.show()



# # Resumen

# ...

# ---
# 
# Los lectores deberán ignorar la siguiente celda.

# En[ ]:




# En[ ]:




