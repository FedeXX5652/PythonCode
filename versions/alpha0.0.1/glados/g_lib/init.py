from . import add
from . import recog
from playsound import playsound
def _init_(text1):
    import subprocess
    try:
        f = open("data/apps.txt", "r")
    except:
      print("ERROR IN OPEN FILE")

    lines = f.readlines() 
    count = 0
    a = text1.lower()
    i = 0
    for line in lines:  
        print("Line {}: {}".format(count, line.strip()))
        count = count + 1

        if a=="" or a==" ":
            print("ERROR IN APP SELECTION")
            break

        elif lines[i].find(a)>=0:
            print("IN")
            try:
                f3 = open("data/subproc.txt", "r")
            except:
                print("ERROR IN OPEN FILE")
            lines3=f3.readlines()

            b=lines3[i]
            print("Path to app: "+b)
            subprocess.Popen([b], stdout=f3, stderr=f3, shell=True)
            f3.close()
            playsound("audio/conf.wav")
            break
            
        i += 1

    f.close()

    if i==count:
        print("No app found, want to add?")
        text2 = ""
        k=""
        while text2 == k:
            text2 = recog._recog_(1)
            if text2 == "sí":
                print("Add func")
                add._add_(text1)
                break
            elif text2 == "no":
                print("Exit program with no adding")
                break
            else:
                print("Can you repeat please?")
                