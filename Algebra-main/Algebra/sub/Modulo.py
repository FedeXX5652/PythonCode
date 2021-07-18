# Modulo de vectores

import math

print("--------- Calculadora de Módulo de Vectores ---------")

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

def modulo2vecs():
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    product = pow(x_value, 2) + pow(y_value, 2)
    module = math.sqrt(product)
    print("producto escalar: ", product)
    print("módulo: ", module)
def modulo3vecs():
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    z_value = int(input("Z = "))
    product = pow(x_value, 2) + pow(y_value, 2) + pow(z_value, 2)
    module = math.sqrt(product)
    print("producto escalar: ", product)
    print("módulo ", module)


if vector_dimensions == 2:
    modulo2vecs()
elif vector_dimensions == 3:
    modulo3vecs()
else:
    print("Dimension Inválida")
