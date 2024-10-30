import json
import networkx as nx
import matplotlib.pyplot as plt


with open("./extracted_relationships.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

relationships = []

for d in data:
    relationships.extend(d["entities"])

# Create a directed graph
G = nx.DiGraph()

# Add edges with labels
for relationship in relationships:
    source = relationship["source_entity"]
    target = relationship["target_entity"]
    label = relationship["relationship"]
    G.add_edge(source, target, label=label)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="skyblue",
    font_size=10,
    font_weight="bold",
    arrowsize=20,
)

edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

# Show plot
plt.show()
