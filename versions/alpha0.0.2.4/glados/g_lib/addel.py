from . import audiorep

def _add_(a, dev):    
    import subprocess
    try:
        f2 = open("data/apps.txt", "a")
        f3 = open("data/subproc.txt", "a")
    except:
      print("ERROR IN OPEN APP OR SUBPROC FILE")
      audiorep._error_()
    path=input("Select a path to "+a+".exe: ")
    f2.write(a+"\n")
    f2.close()
    f3.write(path+"\n")
    f3.close()
    audiorep._conf_()
    try:
      subprocess.Popen([path])
    except:
      print("ERROR IN OPEN APP")
      audiorep._error_()
    return 0

def _del_(a, dev):
  try:
    f = open("data/apps.txt", "r")
  except:
    print("ERROR IN OPEN APP FILE")
    audiorep._error_()

  lines = f.readlines()
  count = 0
  a = a.lower()
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
        print("IN to delete")
      break
    i += 1

  f.close()
  if i == count:
    print("NO APP FOUND")
  else:
    f2 = open("data/apps.txt", "w+")
    

    for line in lines:
      if line.strip("\n") != a:
        f2.write(line)
    f2.close()

    if dev == 1:
      print("i =",i)
    f3 = open("data/subproc.txt", "r")

    lines = f3.readlines()
    f3.close()

    del lines[i]

    f4 = open("data/subproc.txt", "w+")

    for line in lines:
      f4.write(line)


    f4.close()
    audiorep._conf_()

  