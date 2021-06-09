Sebastian Raschka, 2017

Aprendizaje automático con Python - Códigos de ejemplo

## Capítulo 9 - Incrustar un modelo de aprendizaje automático en una aplicación web

- Serializar estimadores de scikit-learn ajustados
- Configurar una base de datos SQLite para el almacenamiento de datos
- Desarrollar una aplicación web con Flask
- Nuestra primera aplicación web con Flask
  - Validación y renderizado de formularios
  - Convertir el clasificador de críticas de cine en una aplicación web
- Desplegar la aplicación web en un servidor público
  - Actualizar el clasificador de películas
- Resumen

---

Puedes el código para las aplicaciones web con Flask en los siguientes directorios:

- `1st_flask_app_1/`: Una app web con Flask sencilla
- `1st_flask_app_2/`: `1st_flask_app_1` ampliado con validación y actualización de formularios flexible 
- `movieclassifier/`: El clasificador de películas incrustado en una aplicación web
- `movieclassifier_with_update/`: los mismo que `movieclassifier` pero con una actualización de la base de datos sqlite desde el inicio


Para ejecutar localmente las aplicaciones web, `cd` en el directorio correspondiente (como se muestra a continuación) y ejecuta el script de la aplicación principal, por ejemplo,

    cd ./1st_flask_app_1
    python3 app.py

Ahora, deberías ver algo parecido a

     * Running on http://127.0.0.1:5000/
     * Restarting with reloader

en tu terminal.
A continuación, abre un navegador web y escribe la dirección que aparece en tu terminal (del tipo http://127.0.0.1:5000/) para ver la aplicación web.


**Enlace a un aplicación de ejemplo creada con este tutorial: http://raschkas.pythonanywhere.com/**.
