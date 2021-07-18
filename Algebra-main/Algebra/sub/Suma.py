# Suma de vectores

import math

print("--------- Calculadora de vector suma ---------")

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

if vector_dimensions == 2:
    vx_value = int(input("Coordenada X del primer vector = "))
    vy_value = int(input("Coordenada Y del primer vector = "))
    wx_value = int(input("Coordenada X del segundo vector = "))
    wy_value = int(input("Coordenada Y del segundo vector = "))
    vecsumx = vx_value + wx_value
    vecsumy = vy_value + wy_value
    vecsumxstr = str(vecsumx)
    vecsumystr = str(vecsumy)
    print("[" + vecsumxstr + ", " + vecsumystr + "]")

elif vector_dimensions == 3:
    vx_value = int(input("Coordenada X del primer vector = "))
    vy_value = int(input("Coordenada Y del primer vector = "))
    vz_value = int(input("Coordenada Z del primer vector = "))
    wx_value = int(input("Coordenada X del segundo vector = "))
    wy_value = int(input("Coordenada Y del segundo vector = "))
    wz_value = int(input("Coordenada Z del segundo vector = "))
    vecsumx = vx_value + wx_value
    vecsumy = vy_value + wy_value
    vecsumz = vz_value + wz_value
    vecsumxstr = str(vecsumx)
    vecsumystr = str(vecsumy)
    vecsumzstr = str(vecsumz)
    print("Vector suma: [" + vecsumxstr + ", " + vecsumystr + ", " + vecsumzstr + "]")
else:
    print("Dimension Inválida")