f = open("input.txt", "r")

lines = f.readlines() 
count = 0
a = "visual"
i = 0
for line in lines:
    print("Line {}: {}".format(count, line.strip()))
    count = count + 1

    if a=="" or a==" ":
        print("ERROR IN APP SELECTION")
        break

    elif lines[i].find(a)>=0:
        print("IN")
        break
    i += 1

f.close()

if i==count:
    print("ERROR")
    f2 = open("input.txt", "a")
    f2.write("\n"+a)
    f2.close()    
