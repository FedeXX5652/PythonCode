def func2(arr):
    value = 10000
    arr2=arr.copy()
    t=r=index=0
    used=[]

    for x in arr2:
        if t<=value:
            t+=x
            used.append(x)
        elif t>value:
            r=t-value
            t=t-r
        
            print("Total: "+str(t))
            print("\n")
            print("Resto: "+str(r))
            print("\n")
            print("Usados: "+str(used))
            print("\n")
            print("-----------------------------")

            fill(str(t), str(r), str(used))

            index+=len(used)
            arr2.append(r)
            used.clear()
            t=0

    # print("Index: "+str(index)+" Len final: "+str(len(arr2)))
    # print("\n")
    # print("Lista final: "+str(arr2))
    # print("\n")
    return(arr2[index::])

def func(arr):
    
    # arr = list(np.random.randint(500, high = 1001, size = 50))
    # print("\n")
    # print("Valores iniciales: "+str(arr))
    # print("\n")
    # print("-----------------------------")

    f= open("COMBINACIÓN_BILLETES.txt","a")
    f.truncate(0)
    f.close()

    now = func2(arr)
    last = []
    while last != now:
        last = now
        now = func2(now)

    res = 0
    if last == now:
        print("No usados: "+str(now))
        print("\n")
        print("Restos: "+str(now[::]))
        print("\n")
        for x in now:
            res +=x
        print("Suma de restos: "+str(res))
        f= open("COMBINACIÓN_BILLETES.txt","a")
        f.write("No usados: "+str(now)+"\n")
        f.write("Suma de no usados: "+str(res))
        f.close()


def fill(t, r, used):
    f= open("COMBINACIÓN_BILLETES.txt","a")

    f.write("Fajo: "+t+"\n")
    f.write("Resto: "+r+"\n")
    f.write("Usados: "+used+"\n")
    f.write("---------------------------------------------"+"\n")

    f.close()

#------------------------------------------------------------------------------------------------------------
import numpy as np

arr = list(np.random.randint(500, 1000, 30))

func(arr)