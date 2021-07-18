# Calculadora de producto escalar de un vector consigo mismo

import math

print("--------- Calculadora de producto escalar de un vector consigo mismo ---------")

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

if vector_dimensions == 2:
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    producto = pow(x_value, 2) + pow(y_value, 2)
    print("producto escalar: ", producto)
elif vector_dimensions == 3:
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    z_value = int(input("Z = "))
    producto = pow(x_value, 2) + pow(y_value, 2) + pow(z_value, 2)
    print("producto escalar: ", producto)
else:
    print("Dimension Inválida")
