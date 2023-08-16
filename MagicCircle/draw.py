import scipy
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams["figure.autolayout"] = True

G = nx.Graph()
nx.circular_layout(G)
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
    

graph(G,[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

plt.figure()
edge_color = [edge_color_func(n1, n2) for n1, n2 in G.edges()]
pos = nx.circular_layout(G)
pos = {n: pos[n] for n in sorted(G.nodes(), key=lambda n: n)}
nx.draw(G, node_color="black", edge_color=edge_color, width=1, with_labels=False, font_color="white", pos=pos, connectionstyle="arc3,rad=0.4")
plt.show()
