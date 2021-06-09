from time import sleep
import keyboard

while keyboard.is_pressed('q') == False:
    print("process 1")
    sleep(5)