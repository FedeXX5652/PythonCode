preguntas = ["Las ballenas son mamiferos?", "La tierra gira al rededor del sol", "Los pulpos son mamiferos"]
respuestas = ["Si", "Si", "No"]

while True:      #Lo repite infinitas veces
    accion = input("Que decea hacer?\n 1. Quiz\n 2. Añadir preguntas\n 3. Ver el quiz\n Ingrese el numero: ")
    print("\n")

    if accion == "1": #Accion 1
        puntos = 0
        for x in range(len(preguntas)): #tomo el tamaño de la lista, osea la cantidad de preguntas y respuestas totales
            print(preguntas[x])
            respuesta_ingresada = input()

            if respuesta_ingresada.lower() == respuestas[x].lower(): #si la respuresta ingresada coincide con la respuesta de la preunta X en la lista
                print("Correcto!"+ "\n")
                puntos += 1
            elif respuesta_ingresada.lower() != respuestas[x].lower(): #si la respuresta ingresada NO coincide con la respuesta de la preunta X en la lista
                print("Incorrecto, la respuesta era: "+respuestas[x].lower())
                print("Respuesta ingresada: "+respuesta_ingresada+ "\n")
            x+=1
        
        print("Puntaje total: "+str(puntos))
    
    elif accion == "2": #Accion 2
        #Recibo la nueva pregunta con su respuesta
        pregunta_nueva = input("Ingrese la nueva pregunta: ")
        respuesta_nueva = input("Ingrese la respuesta: ")

        #Las añado a sus respectivas listas
        preguntas.append(pregunta_nueva)
        respuestas.append(respuesta_nueva)

    elif accion == "3":#Accion 3
        for x in range(len(preguntas)): #tomo el tamaño de la lista, osea la cantidad de preguntas y respuestas totales
            print("Pregunta "+x+": "+preguntas[x]) #muestro las listas
            print("Respuresta de "+x+": "+respuestas[x]+"\n")
    
    else: #Acciones restantes
        print("Ingrese una acción valida")

    print("-------------------------------------------------------------------")


