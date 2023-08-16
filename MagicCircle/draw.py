import scipy
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams["figure.autolayout"] = True

G = nx.Graph()
n = 12
G.add_nodes_from([x+1 for x in range(n)])

def add_edges(G: nx.Graph, conn: int):
    for i in range(n):
        G.add_edge(i+1, (i+conn)%n+1)

def add_edges2(G: nx.Graph, step: int, node: int):
    G.add_edge(node+1, (node+step)%n+1)

def graph(G,mat: list):
    for i in range(mat.__len__()):
        mat[i].reverse()
        for j in range(mat[i].__len__()):
            if mat[i][j] == 1:
                add_edges2(G, i+1, j)

def edge_color_func(n1, n2):
    d1 = abs(n1 - n2)
    d2 = abs(n - abs(n1 - n2))
    distance = min(d1, d2)
    colors = ["black", "blue", "green", "red", "yellow", "orange", "purple", "pink"]
    return colors[distance-1]


def set_level(level: int):
    ns = [1 if x == level else 0 for x in range(n-1)]
    ns.reverse()
    print(ns)
    return ns

def set_school(school: str, mat: list):
    pass

def set_dmg_type(dmg_type: str, mat: list):
    pass

def set_area_type(area_type: str, mat: list):
    pass

def set_range(range_: str, mat: list):
    pass
    
def props_to_mat(level:int, school:str, dmg_type: str, area_type: str, range_: str):
    mat = [[0 for x in range(n-1)] for y in range(n-1)]
    mat[0] = set_level(level)
    # mat[1] = set_school(school, mat[1])
    # mat[2] = set_dmg_type(dmg_type, mat[2])
    # mat[3] = set_area_type(area_type, mat[3])
    # mat[4] = set_range(range_, mat[4])
    print(mat)
    return mat


graph(G,props_to_mat(level=1, school="evocation", dmg_type="cold", area_type="circle", range_="100ft"))

plt.figure()
edge_color = [edge_color_func(n1, n2) for n1, n2 in G.edges()]
pos = nx.kamada_kawai_layout(G)
pos = {n: pos[n] for n in sorted(G.nodes(), key=lambda n: n)}
nx.draw(G, node_color="black", edge_color=edge_color, width=1, with_labels=False, font_color="white", pos=pos, connectionstyle="arc3,rad=-0.4", arrows=True)
plt.show()
