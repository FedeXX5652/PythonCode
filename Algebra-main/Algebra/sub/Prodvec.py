# Producto escalar entre dos vectores

import math

print("--------- Calculadora de producto escalar entre dos vectores ---------")

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

if vector_dimensions == 2:
    vx_value = int(input("Coordenada X del primer vector = "))
    vy_value = int(input("Coordenada Y del primer vector = "))
    wx_value = int(input("Coordenada X del segundo vector = "))
    wy_value = int(input("Coordenada Y del segundo vector = "))
    producto = (vx_value*wx_value) + (vy_value*wy_value)
    print("Producto escalar:", producto)
elif vector_dimensions == 3:
    vx_value = int(input("Coordenada X del primer vector = "))
    vy_value = int(input("Coordenada Y del primer vector = "))
    vz_value = int(input("Coordenada Z del primer vector = "))
    wx_value = int(input("Coordenada X del segundo vector = "))
    wy_value = int(input("Coordenada Y del segundo vector = "))
    wz_value = int(input("Coordenada Z del segundo vector = "))
    producto = (vx_value*wx_value) + (vy_value*wy_value) + (vz_value*wz_value)
    print("Producto escalar:", producto)
else:
    print("Dimension Inválida")
