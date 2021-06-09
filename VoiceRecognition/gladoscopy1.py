import pyaudio
import speech_recognition as sr
import keyboard
import os
import webbrowser
from googlesearch import search
from playsound import playsound
import subprocess


chromedir= 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'

def _recog_():
    r = sr.Recognizer()
    text1 = ""
    #playsound("audio/glados-welcome.mp3")
    with sr.Microphone() as source:
    
        print("Say Something...")
        r.adjust_for_ambient_noise(source)
        audio = r.record(source, duration=3)    #for limited time in seconds#
        #audio = r.listen(source)                #for ilimited listen time#

        try:
            text = r.recognize_google(audio, language='es-AR')
            text_oo = r.recognize_google(audio,show_all=True,language='es-AR')
            print("What did you say: {}".format(text))
            print("Other options: {}".format(text_oo))
            print("")
            text1 = str(text)

        except:
            #playsound("audio/didyousaysomething.mp3")
            print("I am sorry! I can not understand!")
            print("")
            text1 = ""
    return text1




def _action_(text1):

    
    k= text1.lower().split()

    if "buscar" in k[0]:
    	print("Searching "+k[1]+" in Google")

    elif "abrir" in k[0] or len(k)==1 and k[0]!="":
       print("Initializing app....")
       if len(k)==1:
           _init_(k[0])
       else:
           _init_(k[1])
    elif "agregar" in k[0]:
       print("Adding app")
       _add_(k[1])
    elif k=="" or k==" ":
       print("ERROR IN APP SELECTION")
    
    else:
       print("Other action"+"\n")
        



def _init_(text1):
    f = open("data/apps.txt", "r")

    lines = f.readlines() 
    count = 0
    a = text1.lower()
    i = 0
    for line in lines:  
        print("Line {}: {}".format(count, line.strip()))
        count = count + 1

        if a=="" or a==" ":
            print("ERROR IN APP SELECTION")
            print("Adding app")
            break

        elif lines[i].find(a)>=0:
            print("IN")
            f3 = open("data/subproc.txt", "r")
            lines3=f3.readlines()

            b=lines3[i]
            print("Path to app: "+b)
            subprocess.Popen([b], stdout=f3, stderr=f3, shell=True)
            f3.close()
            break
            
        i += 1

    f.close()

    if i==count:
        _add_(a)
        

def _add_(a):
    print("ADDING")
    f2 = open("data/apps.txt", "a")
    f3 = open("data/subproc.txt", "a")

    path=input("Select a path to "+a+".exe: ")

    f2.write("\n"+a)
    f2.close()
    f3.write("\n"+path)
    f3.close()
    subprocess.Popen([path])


def _web_find_(query):
    print("This is what I foud for "+query+":")
    for j in search(query, tld="com", lang='en', num=1, stop=3, pause=2): 
        print(j)
    

while 1:
    shot_pressed = 0
    was_pressed = False
    
    
    while True:
        if keyboard.is_pressed('+'):
            if not was_pressed:
                shot_pressed += 1
                was_pressed = True


        else:
            was_pressed = False


        if was_pressed==True:
                    
            text1 = _recog_()

            if text1 != "":
                _action_(text1)