from twisted.web.server import Site
from twisted.web.resource import Resource
from twisted.internet import reactor, endpoints


class FormPage(Resource):
    def render_GET(self, request):
        return b"""\
            <!DOCTYPE html>
            <html>
            <head><meta charset="utf-8"></head>
            <body>
                <h1>Inscripci&oacute;n de alumno</h1>
                <form method="post">
                    Alumno:<br>
                    <input type="text" name="student">
                    <br>
                    Fecha:<br>
                    <input type="date" name="date">
                    <br>
                    Nombre del curso:<br>
                    <input type="text" name="course">
                    <br><br>
                    <input type="submit" value="Enviar">
                </form> 
            </body>
            </html>\
        """

    def render_POST(self, request):
        if all((request.args[f][0] for f in (b"student", b"date", b"course"))):
            output = "¡Alumno inscripto correctamente!"
        else:
            output = "Complete todos los campos."
        return """\
            <!DOCTYPE html>
            <html>
            <head><meta charset="utf-8"></head>
            <body>{}</body>
            </html>\
        """.format(output).encode("utf-8")


root = Resource()
root.putChild(b"form", FormPage())
factory = Site(root)
endpoint = endpoints.TCP4ServerEndpoint(reactor, 8880)
endpoint.listen(factory)
reactor.run()