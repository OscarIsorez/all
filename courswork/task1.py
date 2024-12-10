import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Load protein sequences
file_path = "data/sequences.txt.xz"
sequences = []

try:
    with lzma.open(file_path, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                seq_record = SeqRecord(Seq(line), id=f"seq{i+1}")
                sequences.append(seq_record)
except Exception as e:
    print(f"Error reading file: {e}")

if not sequences:
    print("No sequences loaded. Please check the file path and content.")
    exit()
else:
    print(f"Loaded {len(sequences)} sequences.")

aligner = PairwiseAligner()
aligner.mode = "local"

n = len(sequences)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        score = aligner.score(sequences[i].seq, sequences[j].seq)
        max_len = max(len(sequences[i].seq), len(sequences[j].seq))
        dist_matrix[i, j] = 1 - (score / max_len)
        dist_matrix[j, i] = dist_matrix[i, j]


plot = plt.imshow(dist_matrix, cmap="viridis")
plt.colorbar(plot)
plt.title("Sequence Similarity Matrix")
plt.savefig("results/t1_matrix.png", dpi=250, bbox_inches="tight")

cutoff = 0.4
G = nx.Graph()

for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] <= cutoff:
            G.add_edge(i, j)

print(
    f"Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
)


degrees = [val for (node, val) in G.degree()]
avg_clustering = nx.average_clustering(G)

print(f"Average Degree: {np.mean(degrees):.2f}")
print(f"Average Clustering Coefficient: {avg_clustering:.2f}")


dbscan = DBSCAN(eps=0.4, min_samples=4, metric="precomputed")
labels = dbscan.fit_predict(dist_matrix)
unique_labels = np.unique(labels)
print(f"Unique labels after parameter adjustment: {unique_labels}")

if len(unique_labels) > 1:
    silhouette = silhouette_score(dist_matrix, labels, metric="precomputed")
    print(f"DBSCAN Silhouette Score: {silhouette:.2f}")
else:
    print("DBSCAN did not find more than one cluster.")

# Try Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average")
labels = agglo.fit_predict(dist_matrix)
silhouette = silhouette_score(dist_matrix, labels, metric="precomputed")
print(f"Agglomerative Clustering Silhouette Score: {silhouette:.2f}")


pos = nx.spring_layout(G, seed=42)
colors = [labels[node] if node < len(labels) else -1 for node in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw(G, pos, node_color=colors, with_labels=False, cmap=plt.cm.rainbow, node_size=50)
plt.title("Sequence Similarity Network")
plt.savefig("results/network.png", dpi=250, bbox_inches="tight")
plt.show()

# plot of a table with statistics of network topological properties
plt.figure(figsize=(8, 4))
plt.axis("off")
plt.table(
    cellText=[[f"Nodes: {G.number_of_nodes()}", f"Edges: {G.number_of_edges()}"]],
    cellLoc="center",
    loc="center",
)
plt.title("Network Statistics")
plt.savefig("results/t1_network_stats.png", dpi=250, bbox_inches="tight")
plt.show()
