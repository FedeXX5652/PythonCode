# Angulo entre vectores y perpendicularidad

print("--------- Calculadora de angulo entre vectores y perpendicularidad ---------")

import math
import math2

vecdim = math2.prompt4dim()

if vecdim == 2:
    vx = math2.get_vx_value()
    vy = math2.get_vy_value()
    wx = math2.get_wx_value()
    wy = math2.get_wy_value()
    prod = math2.producto2(vx, vy, wx, wy)
    mod = math.sqrt(math2.modulo2(vx, vy)*math2.modulo2(wy, wx))
    print(vx, vy, wx, wy)
    print("prod: ", prod)
    print("mod ", mod)
    ang = math.degrees(math.acos(prod/mod))
    perpen = math2.checkperpen(prod)
    print("Angulo: ", ang,"°")
    if perpen == True:
        print("Los vectores son perpendiculares")
    elif perpen == False:
        print("Los vectores no son perpendiculares")

if vecdim == 3:
    vx = math2.get_vx_value()
    vy = math2.get_vy_value()
    vz = math2.get_vz_value()
    wx = math2.get_wx_value()
    wy = math2.get_wy_value()
    wz = math2.get_wz_value()
    prod = math2.producto3(vx, vy, vz, wx, wy, wz)
    mod = math.sqrt(math2.modulo3(vx, vy, vz)*math2.modulo3(wy, wx, wz))
    ang = math.degrees(math.acos(prod/mod))
    perpen = math2.checkperpen(prod)
    print("Angulo: ", ang,"°")
    if perpen == True:
        print("Los vectores son perpendiculares")
    elif perpen == False:
        print("Los vectores no son perpendiculares")