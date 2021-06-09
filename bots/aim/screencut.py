import pyautogui
import time
import keyboard
import random
import win32api
import win32con

while 1:
    if pyautogui.locateOnScreen('stickman.png', region=(0,160,130,370), grayscale=True, confidence=0.8) != None:
        print("On screen")
        time.sleep(0.5)
    else:
        print("Off screen")
        time.sleep(0.5)