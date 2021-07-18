# Master Vector
from sub import Suma, Resta, ProdescAM, Prodvec, Prodvecself, Modulo, Distancia, Vecescalar, Angulo

def masterprogram():
    print("Elige la operación a realizar")
    print("1. Suma de vectores")
    print("2. Vector por escalar")
    print("3. Resta de vectores")
    print("4. Producto escalar entre dos vectores")
    print("5. Producto escalar de un vector consigo mismo")
    print("6. Modulo de un vector")
    print("7. Distancia entre dos puntos")
    print("8. Angulo entre vectores y perpendicularidad")
    print("9. Producto escalar a partir de angulo y modulo")
    opcion = int(input("Número de operación: "))
    if opcion == 1:
        Suma.Suma()
    elif opcion == 2:
        Vecescalar.Vecescalar()
    elif opcion == 3:
        Resta.Resta()
    elif opcion == 4:
        Prodvec.Prodvec()
    elif opcion == 5:
        Prodvecself.Prodvecself()
    elif opcion == 6:
        Modulo.Modulo()
    elif opcion == 7:
        Distancia.Distancia()
    elif opcion == 8:
        Angulo.Angulo()
    elif opcion == 9:
        ProdescAM.ProdescAM()
    else:
        print("opción invalida")

masterprogram()

cont = input("Querés realizar otra operación? Si / No : ")

if cont == "Si" or cont == "si":
    masterprogram()
else:
    quit