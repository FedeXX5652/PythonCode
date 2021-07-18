def restax():
    vx_value = int(input("Coordenada X del primer vector = "))
    wx_value = int(input("Coordenada X del segundo vector = "))
    vecsumx = vx_value - wx_value
    return vecsumx
    
def restay():
    vy_value = int(input("Coordenada Y del primer vector = "))
    wy_value = int(input("Coordenada Y del segundo vector = "))
    vecsumy = vy_value - wy_value
    return vecsumy

def restaz():
    vz_value = int(input("Coordenada Z del primer vector = "))
    wz_value = int(input("Coordenada Z del segundo vector = "))
    vecsumz = vz_value - wz_value
    return vecsumz

def modulo2 (x, y):
    product = pow(x, 2) + pow(y, 2)
    return product

def modulo3 (x, y, z):
    product = pow(x, 2) + pow(y, 2) + pow(z, 2)
    return product

def dimcheck():
    vecdim = int(input("Para vectores en el plano, ingrese 2. \nPara vectores en el espacio, ingrese 3. \n"))
    return vecdim

def prompt4dim():
    while True:
        vecdim = dimcheck()
        if vecdim == 2 or vecdim == 3:
            break
        else:
            print("Dimensión inválida, intente de nuevo")
    return vecdim

def get_vx_value():
    vx_value = int(input("Coordenada X del primer vector = "))
    return vx_value

def get_vy_value():
    vy_value = int(input("Coordenada Y del primer vector = "))
    return vy_value

def get_vz_value():
    vz_value = int(input("Coordenada Z del primer vector = "))
    return vz_value

def get_wx_value():
    wx_value = int(input("Coordenada X del segundo vector = "))
    return wx_value

def get_wy_value():
    wy_value = int(input("Coordenada Y del segundo vector = "))
    return wy_value

def get_wz_value():
    wz_value = int(input("Coordenada Z del segundo vector = "))
    return wz_value

def producto2(x1, x2, y1, y2):
    producto = (x1*y1) + (x2*y2)
    return producto

def producto3(x1, x2, x3, y1, y2, y3):
    producto = (x1*y1) + (x2*y2) + (x3*y3)
    return producto

def checkperpen(cosval):
    if cosval == 0:
        return True
    else:
        return False