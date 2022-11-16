from copy import deepcopy
import math

m = []
fil = 10
col = 10
l = [[3, 4], [2, 4], [1, 4], [4, 3], [4, 2], [4, 1]]
rot = math.radians(45)

for f in range(fil):
    m.append([])
    for c in range(col):
        m[f].append("-")

m[4][4] = "R"
m_orig = deepcopy(m)
m_rep = deepcopy(m)

for i in l:
    m[i[0]][i[1]] = "L"


for f in range(fil):
    for c in range(col):
        print(m[f][c], end=" ")
    print("\n")

for f in range(fil):
    for c in range(col):
        print(m_orig[f][c], end=" ")
    print("\n")


dx = 4
dy = 4
for w in range(8):
    m_rep = deepcopy(m_orig)
    for i in range(len(l)):
        x = l[i][0]
        y = l[i][1]
        ox, oy = dx, dy

        qx = ox + math.cos(rot) * (x - ox) + math.sin(rot) * (y - oy)
        qy = oy + -math.sin(rot) * (x - ox) + math.cos(rot) * (y - oy)
        print(int(qx),int(qy))
        m_rep[int(qx)][int(qy)] = "L"
        l[i] = [qx, qy]

    for f in range(fil):
        for c in range(col):
            print(m_rep[f][c], end=" ")
        print("\n")

    print("\n")
    print("\n")