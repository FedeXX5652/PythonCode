from PyQt5 import QtWidgets
import sys


class MyScrollWidget(QtWidgets.QWidget):

    def __init__(self):
        super(MyScrollWidget, self).__init__()
        lay = QtWidgets.QVBoxLayout(self)

        scrollArea = QtWidgets.QScrollArea()
        lay.addWidget(scrollArea)
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout()

        for i in range(10):
            group_box = QtWidgets.QGroupBox()

            layout = QtWidgets.QHBoxLayout(group_box)

            label = QtWidgets.QLabel()
            label.setText('Label For Item {0}'.format(i))
            layout.addWidget(label)

            push_button = QtWidgets.QPushButton(group_box)
            push_button.setText('Run Button')
            push_button.setFixedSize(100, 32)
            layout.addWidget(push_button)

            top_layout.addWidget(group_box)

        top_widget.setLayout(top_layout)
        scrollArea.setWidget(top_widget)
        self.resize(200, 500)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MyScrollWidget()
    widget.show()
    sys.exit(app.exec_())