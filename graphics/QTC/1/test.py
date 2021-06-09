from test_ui import *

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self) 
        self.label.setText("Haz clic en el botón")
        self.pushButton.setText("Presióname")
        self.pushButton.clicked.connect(self.actualizar)
    def actualizar(self):
        self.label.setText("¡Acabas de hacer clic en el botón!")
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
