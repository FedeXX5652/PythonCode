from PyQt5 import QtWidgets, uic
import sys
import os
import json
import draw

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('view.ui', self)

        self.settings = None

        with open(os.path.join(MODULE_DIR, "settings.json"), "r", encoding="utf-8") as f:
            settings:dict = json.load(f)
            lvl_nodes = 1 if settings.get("max_lvl") else 0
            form_nodes = len(settings.get("forms")) if settings.get("forms") else 0
            effects = 0
            for school in settings.get("effects").keys():
                print(school, len(settings.get("effects").get(school)))
                if len(settings.get("effects").get(school)) > effects:
                    effects = (len(settings.get("effects").get(school)))
            self.settings = settings
            
            # n is the highest number of nodes that the graph can have
            n = lvl_nodes if (lvl_nodes > form_nodes) else (form_nodes if (form_nodes > effects) else effects)
            draw.n = n

        self.lvl_input = self.findChild(QtWidgets.QLineEdit, 'lvl_input')
        self.effects_combo_box = self.findChild(QtWidgets.QComboBox, 'effects_combo_box')
        self.effects_table = self.findChild(QtWidgets.QTableWidget, 'effects_table')
        self.selected_effects_table = self.findChild(QtWidgets.QTableWidget, 'selected_effects_table')
        self.dlt_selected_effects_btn = self.findChild(QtWidgets.QPushButton, 'dlt_selected_effects_btn')
        self.forms_table = self.findChild(QtWidgets.QTableWidget, 'forms_table')
        self.selected_forms_table = self.findChild(QtWidgets.QTableWidget, 'selected_forms_table')
        self.dlt_selected_forms_btn = self.findChild(QtWidgets.QPushButton, 'dlt_selected_forms_btn')

        self.startEffectSelector()

        self.show()

    def startEffectSelector(self):
        # set combo box to keys of effects dict
        self.effects_combo_box.addItems(self.settings.get("effects").keys())

        """
            set table to values of the firts key of effects dict
            the data is a list of dicts as {
                    "Efecto": "Caída Lenta",
                    "Regla": "Mantenimiento, Reacción, Poción",
                    "Parámetro": "",
                    "Coste": "NH"
                }
            we need to get the keys of the first dict to set the columns
        """
        self.effects_table.setColumnCount(len(self.settings.get("effects").get(list(self.settings.get("effects").keys()).__getitem__(0)).__getitem__(0).keys()))
        self.effects_table.setHorizontalHeaderLabels(self.settings.get("effects").get(list(self.settings.get("effects").keys()).__getitem__(0)).__getitem__(0).keys())
        self.effects_table.setRowCount(len(self.settings.get("effects").get(list(self.settings.get("effects").keys()).__getitem__(0))))
        for i, effect in enumerate(self.settings.get("effects").get(list(self.settings.get("effects").keys()).__getitem__(0))):
            for j, value in enumerate(effect.values()):
                self.effects_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(value)))

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()