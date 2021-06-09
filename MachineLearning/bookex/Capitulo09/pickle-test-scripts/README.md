**Nota**

La sección de pickling puede ser un poco complicada por lo que he incluido en este directorio unos scripts de prueba más sencillos (pickle-test-scripts/) para comprobar si tu entorno está configurado correctamente. Básicamente, se trata de una versión reducida de las secciones más importantes del capítulo 8, incluyendo un subconjunto movie_review_data muy pequeño.

La ejecución de

    python pickle-dump-test.py

entrenará un pequeño modelo de clasificación desde el `movie_data_small.csv` y creará 2 archivos pickle  

    stopwords.pkl
    classifier.pkl

A continuación, si ejecutas

    python pickle-load-test.py

podrás ver las 2 siguientes líneas como salida:

    Prediction: positive
    Probability: 85.71%