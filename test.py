import networkx as nx
import matplotlib.pyplot as plt

G = nx.MultiGraph()
e = [(1, 2), (1, 2), (2, 3), (3, 4),(3, 4),(3, 4)]  # list of edges
G.add_edges_from(e)

plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams["figure.autolayout"] = True
plt.figure()
plt.axis("off")

pos = nx.spring_layout(G, seed=42)  # Layout for positioning nodes

r = 3
ax = plt.gca()
for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(r+0.3*e[2])),
                                color='red'
                                )
    )

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="black")

# Draw labels
nx.draw_networkx_labels(G, pos, font_color="white")

plt.show()
