from . import addel
from . import init
from . import recog
from . import web_find
from . import audiorep

def _action_(text1, dev):

   
   w= text1.lower()
   try:
      f = open("data/key.txt", "r")
   except:
      print("ERROR IN OPEN KEY FILE")
      audiorep._error_()
      return None
   lines = f.readlines()
   count = 0


   for line in lines:
      if dev == 1:
         print("Line {}: {}".format(count, line.strip()))
      k = line.strip().lower().split()

      for i in range(len(k)):
         if dev == 1:
            print("k{}: {}".format(i, k[i]))
         x = w.find(k[i])
         if x != -1:
            if dev == 1:
               print(x)
            break
         i+=1
      if x != -1:
         EO_for = False
         break
      count = count + 1
      EO_for = True


   if EO_for == False:
      w=w.split()
      if dev == 1:
         print("w: {}".format(w))
      k = k[i]
      if dev == 1:
         print("k: {}".format(k))
      w.remove(k)

   if dev == 1:
      print(w)
   final = ''.join(w)
   if dev == 1:
      print(final)

   if count==0 or EO_for==True:
      if w == []:
         print("Which app do you want to execute?")
         text1 = recog._recog_(1)
         init._init_(text1, dev)
      else:
         print("OPEN")
         init._init_(final, dev)

   elif count==1:
      if w == []:
         print("Which app do you want to add?")
         text1 = recog._recog_(1)
         addel._add_(text1, dev)
      else:
         print("ADD")
         addel._add_(final, dev)

   elif count==2:
      if w == []:
         print("What do you want to search?")
         text1 = recog._recog_(1)
         web_find._web_find_(text1)
      else:
         print("SEARCH")
         web_find._web_find_(final)

   elif count==3:
      if w == []:
         print("Which app do you want to delete?")
         text1 = recog._recog_(1)
         addel._del_(text1, dev)
      else:
         print("DELETE")
         addel._del_(final, dev)
