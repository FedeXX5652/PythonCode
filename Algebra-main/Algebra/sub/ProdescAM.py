# Producto escalar a partir de modulo y angulo

import math

print("--------- Calculadora de producto escalar a partir de modulo y angulo ---------")

print("Desea ingresar el modulo como: ")
print("1. Numero real")
print("2. Raiz cuadrada")

opcion = int(input("Opcion numero: "))

if opcion == 1:
    modv = float(input("Modulo del primer vector: "))
    modw = float(input("Modulo del segundo vector: "))
    ang = math.radians(float(input("Angulo (en grados, sin puntos, comas o simbolos): ")))
    prod = round(modv*modw*math.cos(ang), 2)
    print(prod)
elif opcion == 2:
    modv = round(math.sqrt(float(input("Modulo del primer vector (numero real sobre el cual se le calculará la raiz cuadrada, solo el numero sin ningun simbolo): "))), 2)
    modw = round(math.sqrt(float(input("Modulo del segundo vector (numero real sobre el cual se le calculará la raiz cuadrada, solo el numero sin ningun simbolo): "))), 2)
    ang = round(math.cos(math.radians(float(input("Angulo (en grados, sin puntos, comas o simbolos): ")))), 2)
    prod = round(modv*modw*ang, 2)
    print(prod)