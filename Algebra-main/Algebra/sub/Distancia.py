# Distancia entre dos puntos

print("--------- Calculadora de distancia entre puntos ---------")

import math
import math2

vector_dimensions = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))

if vector_dimensions == 2:
    VecRestaX = math2.restax()
    VecRestaY = math2.restay()
    Modulo = math2.modulo2(VecRestaX, VecRestaY)
    Distancia = round(math.sqrt(Modulo), 2)
    print("La distancia es la raiz cuadrada de ", Modulo, "o", Distancia)
elif vector_dimensions == 3:
    VecRestaX = math2.restax()
    VecRestaY = math2.restay()
    VecRestaZ = math2.restaz()
    Modulo = math2.modulo3(VecRestaX, VecRestaY, VecRestaZ)
    Distancia = round(math.sqrt(Modulo), 2)
    print("La distancia es la raiz cuadrada de ", Modulo, "o", Distancia)
else:
    quit
