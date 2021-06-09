import os
def _close_(a, dev):
    command = "taskkill /f /im {app}.exe"

    try:
        os.system(command.format(app = a))
    except:
        print("No app found")