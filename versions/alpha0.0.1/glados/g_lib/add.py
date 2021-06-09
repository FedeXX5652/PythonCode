def _add_(a):    
    import subprocess

    print("ADDING")
    try:
        f2 = open("data/apps.txt", "a")
        f3 = open("data/subproc.txt", "a")
    except:
      print("ERROR IN OPEN FILE")
    path=input("Select a path to "+a+".exe: ")
    f2.write("\n"+a)
    f2.close()
    f3.write("\n"+path)
    f3.close()
    subprocess.Popen([path])