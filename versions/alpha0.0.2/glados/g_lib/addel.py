def _add_(a):    
    import subprocess

    print("ADDING")
    try:
        f2 = open("data/apps.txt", "a")
        f3 = open("data/subproc.txt", "a")
    except:
      print("ERROR IN OPEN FILE")
    path=input("Select a path to "+a+".exe: ")
    f2.write(a+"\n")
    f2.close()
    f3.write(path+"\n")
    f3.close()
    try:
      subprocess.Popen([path])
    except:
      print("ERROR IN OPEN APP")
    return 0

def _del_(a):
  try:
    f = open("data/apps.txt", "r")
  except:
    print("ERROR IN OPEN FILE")

  lines = f.readlines()
  count = 0
  a = a.lower()
  i = 0
  for line in lines:  
    print("Line {}: {}".format(count, line.strip()))
    count = count + 1

    if a=="" or a==" ":
      print("ERROR IN APP SELECTION")
      break
    elif lines[i].find(a)>=0:
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

    print("i =",i)
    f3 = open("data/subproc.txt", "r")

    lines = f3.readlines()
    f3.close()

    del lines[i]

    f4 = open("data/subproc.txt", "w+")

    for line in lines:
      f4.write(line)


    f4.close()

  