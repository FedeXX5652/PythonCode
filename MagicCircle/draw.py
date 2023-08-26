import operator
import random
import matplotlib.pyplot as plt
import networkx as nx
import os

n = 0

schools = ["abjuration", "conjuration", "divination", "enchantment", "evocation", "illusion", "necromancy", "transmutation"]

dmg_types = ["acid", "bludgeoning", "cold", "damage", "extra", "fire", "force", "lightning", "necrotic", "nonmagical", "piercing", "poison", "psychic", "radiant", "slashing", "thunder"]

def pad(num: str, ns: int = n):
    pad = "0" * (ns - len(num))
    return pad + num

area_types = {"circle":pad("1111"), "cone/sphere": pad("1011"), "cone":pad("1001"),
                  "cube":pad("1"), "cylinder":pad("100011"), "line":pad("11"), "multiple targets/sphere":pad("111"),
                  "multiple targets":pad("101"), "none":pad("100101"), "single target/cone": pad("11011"),
                  "single target/cube":pad("10111"), "single target/multiple targets":pad("11001"), "single target/sphere":pad("11101"),
                  "single target/wall":pad("11111"), "single target":pad("10101"), "sphere/cylinder":pad("10011"),
                  "sphere":pad("10001"), "square":pad("1101"), "wall":pad("100001")}

ranges = ["10ft radius", "100ft line", "15ft cone", "15ft cube",
            "15ft radius", "30ft cone", "30ft line", "30ft radius",
            "5ft radius", "60ft cone", "60ft line", "1mi point",
            "10ft point", "1000ft point", "120ft point", "150ft point",
            "30ft point", "300ft point", "5ft point", "500ft point",
            "60ft point", "90ft point", "self", "sight",
            "special", "touch"]


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

def row_factory(num: int):
    row = [0 for _ in range(n-1)]
    num_bin = bin(num)[2:]
    num_list = [int(x) for x in num_bin]
    
    if len(num_list) > n - 1:
        raise ValueError(f"Input number {num} produces a binary representation longer than n - 1 digits.")
    
    for i in range(len(num_list)):
        row[-(i + 1)] = num_list[-(i + 1)]
    
    row = (row+[1])
    return row

def set_level(level: int):
    return row_factory(level)


def set_school(school: str):
    index = schools.index(school)
    return row_factory(index)


def set_dmg_type(dmg_type: str):
    index = dmg_types.index(dmg_type)
    return row_factory(index)


def set_area_type(area_type: str):
    return [int(x) for x in area_types[area_type]]


def set_range(range_: str):
    index = ranges.index(range_)
    return row_factory(index)

def props_to_mat(level:int=-1, school:str="", dmg_type: str="", area_type: str="", range_: str=""):
    mat = [[] for y in range(5)]
    mat[0] = set_level(level) if level != -1 else [0 for x in range(n+1)]
    mat[1] = set_school(school) if school != "" else [0 for x in range(n+1)]
    mat[2] = set_dmg_type(dmg_type) if dmg_type != "" else [0 for x in range(n+1)]
    mat[3] = set_area_type(area_type) if area_type != "" else [0 for x in range(n+1)]
    mat[4] = set_range(range_) if range_ != "" else [0 for x in range(n+1)]
    return mat

def draw(level, school, dmg_type, area_type, range_, radius, graph_type, node_label, edge_color, node_size):

    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    G = nx.MultiGraph()
    G.add_nodes_from([x+1 for x in range(n)])
    
    # add one pre defined edge
    G.add_edge(2, 4)
    G.add_edge(4, 2)

    graph_matrix = props_to_mat(level=level, school=school, dmg_type=dmg_type, area_type=area_type, range_=range_)
    graph(G, graph_matrix)

    # Encontrar nodos aislados
    isolated_nodes = list(nx.isolates(G))

    # Eliminar nodos aislados del grafo
    G.remove_nodes_from(isolated_nodes)

    plt.figure()
    plt.axis("off")
    
    if graph_type == "circular":
        pos = nx.circular_layout(G)
    elif graph_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif graph_type == "random":
        pos = nx.random_layout(G)
    elif graph_type == "shell":
        pos = nx.shell_layout(G)
    elif graph_type == "planar":
        pos = nx.planar_layout(G)
    elif graph_type == "spring":
        pos = nx.spring_layout(G, k=1)
    else:
        pos = nx.circular_layout(G)
    pos = {n: pos[n] for n in sorted(G.nodes(), key=lambda n: n)}
    
    # nx.draw(G, node_color="black", edge_color=edge_color, width=1, with_labels=node_label, font_color="white", pos=pos, arrows=True, connectionstyle=f"arc3,rad={radius}",
    #         node_size=node_size)
    

    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle=f"arc3,rad={radius+0.3*e[2]}",
                                    color=edge_color_func(e[0], e[1]) if edge_color else "black",
                                    )
        )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="black")

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_color="white") if node_label else None


    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(MODULE_DIR, "fig.png")):
        os.remove(os.path.join(MODULE_DIR, "fig.png"))
    plt.savefig("fig.png")
    plt.close()