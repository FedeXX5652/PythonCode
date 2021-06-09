# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GLaDOS.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("TEST 1")
        MainWindow.resize(1109, 689)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("border-image: url(../templates/main_background.jpg) 0 0 0 0 stretch stretch;\n""font: 75 18pt \"font\";")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 70, 371, 171))
        self.pushButton.setStyleSheet("border-image: url(../templates/main_border.jpg);\n""font: 75 30pt \"Ebrima\"rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 350, 100, 50))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1109, 40))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(lambda: self.clicked("EVENT"))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TEST01"))
        self.pushButton.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">HOLA</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "BUTTON 1"))
        
        self.label.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">HOLA</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "LABEL 1"))


    def clicked(self, text):
        self.label.setText(text)
        self.label.adjustSize()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

