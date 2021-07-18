# Vector por Escalar

import math

print("--------- Calculadora de vector por un escalar ---------")

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

if vector_dimensions == 2:
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    escalar = int(input("Escalar = "))
    vectorx = x_value*escalar
    vectory = y_value*escalar
    vectorxstr = str(vectorx)
    vectorystr = str(vectory)
    print("Producto escalar: [" + vectorxstr + ", " + vectorystr + "]")
elif vector_dimensions == 3:
    x_value = int(input("X = "))
    y_value = int(input("Y = "))
    z_value = int(input("Z = "))
    escalar = int(input("Escalar = "))
    vectorx = x_value*escalar
    vectory = y_value*escalar
    vectorz = z_value*escalar
    vectorxstr = str(vectorx)
    vectorystr = str(vectory)
    vectorzstr = str(vectorz)
    print("Producto escalar: [" + vectorxstr + ", " + vectorystr + ", " + vectorzstr +"]")
else:
    print("Dimension Inválida")
