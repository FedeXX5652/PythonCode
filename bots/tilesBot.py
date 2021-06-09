import pyautogui
import time
import keyboard
import random
import win32api
import win32con

"""
tile 1: X:  734 Y:  824 RGB: ( 81,  84, 116)
tile 2: X:  837 Y:  823 RGB: ( 87,  89, 117)
tile 3: X: 1014 Y:  833 RGB: ( 80,  83, 116)
tile 4: X: 1130 Y:  824 RGB: ( 81,  85, 116)
"""


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

print("work")
while keyboard.is_pressed('q') == False:
    
    if pyautogui.pixel(734, 824)[0] == 0:
        print("1")
        click(734, 824)

    if pyautogui.pixel(837, 824)[0] == 0:
        print("2")
        click(837, 824)

    if pyautogui.pixel(1014, 824)[0] == 0:
        print("3")
        click(1014, 824)

    if pyautogui.pixel(1130, 824)[0] == 0:
        print("4")
        click(1130, 824)