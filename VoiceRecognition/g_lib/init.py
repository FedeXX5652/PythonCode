from . import addel
from . import recog
from . import audiorep

def _init_(text1, dev):
    import subprocess
    import os
    try:
        f = open("data/apps.txt", "r")
    except:
      print("ERROR IN OPEN FILE")
      audiorep._error_()

    lines = f.readlines() 
    count = 0
    a = text1.lower()
    i = 0
    for line in lines:
        if dev == 1:
            print("Line {}: {}".format(count, line.strip()))
        count = count + 1

        if a=="" or a==" ":
            print("ERROR IN APP SELECTION")
            break

        elif lines[i].find(a)>=0:
            if dev == 1:
                print("IN")
            try:
                f3 = open("data/subproc.txt", "r")
            except:
                print("ERROR IN OPEN SUBPROC FILE")
                audiorep._error_()
            lines3=f3.readlines()
            if dev == 0:
                print("Line {}: {}".format(count, line.strip()))
            b=lines3[i]
            print("Path to app: "+b)


            s = b.strip().replace('\\\\', "\\")

            if dev == 1:
                print(os.path.exists(s))
            if os.path.exists(s)==True:
                subprocess.Popen([b], stdout=f3, stderr=f3, shell=True)
                audiorep._conf_()
            else:
                print("ERROR IN APP PATH")
                audiorep._error_()
            f3.close()
            
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
                if dev == 1:
                    print("Add func")
                addel._add_(text1, dev)
                break
            elif text2 == "no":
                print("Exit program with no adding")
                break
            else:
                print("Can you repeat please?")
                