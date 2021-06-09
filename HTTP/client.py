import requests
url = "http://localhost:5000/alumno"

while True:
    print("""
        \nAdministracion de alumnos
        ---------------------------
        1. agregar
        2. modificar
        3. listar
        4. eliminar
        5. salir

        """)
    opcion = input("Selenccione una opción: ")

    if opcion == "1":
        nombre = input("Ingrese el nombre: ")
        cursos = input("Ingrese los cursos: ")
        datos = {"nombre":nombre, "cursos":cursos}
        r = requests.post(url, json=datos)
        print("Server code response: ", r.status_code)
        print("Server content response: ", r.json())
    
    elif opcion == "2":
        while True:
            try:
                id_alumno = int(input("ID de alumno a modificar: "))
                break
            except ValueError:
                print("Ingrese un ID valido")
        
        datos = {"id":id_alumno, "nombre": None, "cursos":None}

        cambio = input("Desea modificar nombre (y/N)? ")
        if cambio.casefold() == "y":
            datos['nombre'] = input("Ingrese nuevo nombre: ")
        
        cambio = input("Desea modificar cursos (y/N)? ")
        if cambio.casefold() == "y":
            datos['cursos'] = input("Ingrese nueva cantidad de cursos: ")
        
        r = requests.put(url, json=datos)
        print("Server code response: ", r.status_code)
        print("Server content response: ", r.json())
        
    elif opcion == "3":
        r = requests.get(url)
        if r.status_code == 200:
            print("Lista de alumnos: ")
            print("------------------")
            for alumno in r.json()["alumnos"]:
                print(alumno)
        else:
            print("No se pudo imprimir la lista de alumnos")
            print("Server code response: ", r.status_code)

    elif opcion == "4":
        while True:
            try:
                id_alumno = int(input("ID de alumno a borrar: "))
                break
            except ValueError:
                print("Ingrese un ID valido")
        
        datos = {"id":id_alumno}
        r = requests.delete(url, json=datos)
        print("Server code response: ", r.status_code)
        print("Server content response: ", r.json())

    elif opcion == "5":
        print("Finalizando sesión.......")

    else:
        print("Elija una opcion valida")
