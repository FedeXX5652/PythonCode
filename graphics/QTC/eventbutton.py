from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore
 
i=0
class Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        title = "PyQt5 Signal And Slots"
        left = 500
        top = 200
        width = 300
        height = 250
        iconName = "icon.png"
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconName))
        self.setGeometry(left,  top, width, height)
        self.CreateButton()
        self.show()
 
 
    def CreateButton(self):
        button = QPushButton("Close Application", self)
        button.setGeometry(QRect(100,100,111,28))
        button.setIcon(QtGui.QIcon("icon.png"))
        button.setIconSize(QtCore.QSize(40,40))
        button.clicked.connect(self.ButtonAction)
 
    
    def ButtonAction(self):
        global i
        print("event "+str(i))
        i=i+1
 
 
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())