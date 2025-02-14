import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
import igraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
import seaborn as sns

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

threshold = 0.45

edges = []
weights = []
for i in range(len(dist_matrix)):
    for j in range(i + 1, len(dist_matrix)):
        if dist_matrix[i, j] <= threshold:
            edges.append((i, j))
            weights.append(1 - dist_matrix[i, j])

graph = igraph.Graph(edges=edges, directed=False)
graph.vs["name"] = [f"Seq {i}" for i in range(len(dist_matrix))]

clusters = graph.community_multilevel()
graph.vs["cluster"] = clusters.membership

num_clusters = len(clusters)
palette = sns.color_palette("hsv", num_clusters)
node_colors = [palette[cluster] for cluster in graph.vs["cluster"]]
graph.vs["color"] = [palette[cluster] for cluster in graph.vs["cluster"]]

graph.es["width"] = [8 * weight for weight in weights]  

layout = graph.layout("circle")

igraph.plot(
    graph,
    "results/t1_clustered_network.png",
    layout=layout,
    bbox=(1920, 1440),
    margin=40,
    vertex_label=[name[-3:] for name in graph.vs["name"]],
    vertex_size=20,
    vertex_color=graph.vs["color"],
    edge_width=graph.es["width"],
    edge_color="gray",
)

diameter = graph.diameter()
girth = graph.girth() if graph.girth() != float("inf") else "Infinity"
radius = graph.radius()
avg_path_length = graph.average_path_length()
density = graph.density()
assortativity = graph.assortativity_degree()
clustering_coeff_avg = graph.transitivity_avglocal_undirected()
clustering_coeff_global = graph.transitivity_undirected()

degrees = graph.degree()
avg_degree = np.mean(degrees)
degree_std = np.std(degrees)

metrics = {
    "Diameter": diameter,
    "Girth": girth,
    "Average Path Length": avg_path_length,
    "Density": density,
    "Assortativity": assortativity,
    "Average Degree": avg_degree,
    "Degree Std Dev": degree_std,
    "Avg Clustering Coefficient": clustering_coeff_avg,
    "Global Clustering Coefficient": clustering_coeff_global,
    "Density": density,
}

print("Network topology metrics:\n")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

fig, ax = plt.subplots(figsize=(6, len(metrics) * 0.4))
ax.axis("tight")
ax.axis("off")
table = ax.table(
    cellText=metrics_df.values,
    rowLabels=metrics_df.index,
    colLabels=metrics_df.columns,
    cellLoc="center",
    loc="center",
)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig("results/t1_network_topology_metrics.png")
plt.close()
