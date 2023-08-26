from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('view.ui', self)

        self.button = self.findChild(QtWidgets.QPushButton, 'testBtn')
        self.button.clicked.connect(self.clickMethod)

        self.show()

    def clickMethod(self):
        print('Clicked Pyqt button.')

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()