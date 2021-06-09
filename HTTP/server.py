from flask import Flask, jsonify, request

app = Flask(__name__)

alumnos = []

@app.route('/')
def home():
    return "HOME"

@app.route('/alumno', methods=['GET','POST','PUT','DELETE'])
def alumno():
    
    if request.method == "GET":
        return jsonify({'alumnos': alumnos})
    
    elif request.method == "POST":
        if not alumnos:
            codigo = 1
        else:
            codigo = alumnos[-1]['id'] + 1
        alumno = {
                'id':codigo,
                'nombre':request.json['nombre'],
                'cursos':request.json['cursos'],
                }
        alumnos.append(alumno)
        return jsonify ("Alumno añadido")
    
    elif request.method == "PUT":
        id_alumno = request.json['id']
        for alumno in alumnos:
            if id_alumno == alumno.get('id'):
                if request.json['nombre'] is not None:
                    alumno['nombre'] = request.json['nombre']
                if request.json['cursos'] is not None:
                    alumno['cursos'] = request.json['cursos']
                return jsonify("Datos modificados")
        return jsonify("ID de alumno no encontrado")
    
    elif request.method == "DELETE":
        id_alumno = request.json['id']
        for alumno in alumnos:
            if id_alumno == alumno.get('id'):
                alumnos.remove(alumno)
                return jsonify("Alumno borrado")
        return jsonify("ID de alumno no encontrado")  

@app.route('/alumno/<int:i>')
def student(i):
    try:
        return jsonify({'alumnos': alumnos[i-1]})
    except:
        return jsonify("Alumno no encontrado")

@app.route('/admin')
def admin():
    return jsonify("NO TE METAS CON EL ADMIN")

if __name__ == '__main__':
    app.run(debug=True)