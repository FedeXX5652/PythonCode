"""
 ██████╗ ██╗      █████╗ ██████╗  ██████╗ ███████╗
██╔════╝ ██║     ██╔══██╗██╔══██╗██╔═══██╗██╔════╝
██║  ███╗██║     ███████║██║  ██║██║   ██║███████╗
██║   ██║██║     ██╔══██║██║  ██║██║   ██║╚════██║
╚██████╔╝███████╗██║  ██║██████╔╝╚██████╔╝███████║
 ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚══════╝
    ╔═╗╔═╗╔═╗╦╔═╗╔╦╗  ╔═╗╦ ╦╔═╗╔╦╗╔═╗╔╦╗
    ╠═╣╚═╗╚═╗║╚═╗ ║   ╚═╗╚╦╝╚═╗ ║ ║╣ ║║║
    ╩ ╩╚═╝╚═╝╩╚═╝ ╩   ╚═╝ ╩ ╚═╝ ╩ ╚═╝╩ ╩
"""


# ______       _     __   ____   __
# |  ___|     | |    \ \ / /\ \ / /
# | |_ ___  __| | ___ \ V /  \ V / 
# |  _/ _ \/ _` |/ _ \/   \  /   \ 
# | ||  __/ (_| |  __/ /^\ \/ /^\ \
# \_| \___|\__,_|\___\/   \/\/   \/
                                 
                                 

import os
import sys
from playsound import playsound
from g_lib import recog
from g_lib import action
from g_lib import web_find


print(" ██████╗ ██╗      █████╗ ██████╗  ██████╗ ███████╗")
print("██╔════╝ ██║     ██╔══██╗██╔══██╗██╔═══██╗██╔════╝")
print("██║  ███╗██║     ███████║██║  ██║██║   ██║███████╗")
print("██║   ██║██║     ██╔══██║██║  ██║██║   ██║╚════██║")
print("╚██████╔╝███████╗██║  ██║██████╔╝╚██████╔╝███████║")
print(" ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚══════╝")
print("    ╔═╗╔═╗╔═╗╦╔═╗╔╦╗  ╔═╗╦ ╦╔═╗╔╦╗╔═╗╔╦╗")
print("    ╠═╣╚═╗╚═╗║╚═╗ ║   ╚═╗╚╦╝╚═╗ ║ ║╣ ║║║")
print("    ╩ ╩╚═╝╚═╝╩╚═╝ ╩   ╚═╝ ╩ ╚═╝ ╩ ╚═╝╩ ╩")



f = open("data/log.txt", "r")
lines = f.readlines()
count = 0
i=0
for line in lines:
    #print("Line {}: {}".format(count, line.strip()))
    count = count + 1
    if lines[i].find("dev")>=0:
        dev = lines[i]
        break
    i =+ 1

dev = dev.rstrip("\n")
dev = dev.split(' ')

strings = [str(integer) for integer in dev[1:]]
a_string = "".join(strings)
dev = int(a_string)
if dev == 0:
    playsound("audio/welcome.wav")
else:
    print("Developer version ON")

text1 = ""

while 1:
   while not "glados" in text1:
        text1=recog._recog_(0)
   if "glados" in text1:
        k=""
        text1 = recog._recog_(1)
        while text1 == k:
            text1 = recog._recog_(1)
        if text1 != "":
            action._action_(text1, dev)

        text1 = ""
   print("re init")