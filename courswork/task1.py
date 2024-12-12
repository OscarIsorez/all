import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
import igraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns


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
# Define a similarity threshold


# Threshold for similarity
threshold = 0.45

# Create edge list with weights (based on distances)
edges = []
weights = []
for i in range(len(dist_matrix)):
    for j in range(i + 1, len(dist_matrix)):
        if dist_matrix[i, j] <= threshold:
            edges.append((i, j))
            # Weight is inverse of distance for better visualization
            weights.append(1 - dist_matrix[i, j])

# Create the graph
graph = igraph.Graph(edges=edges, directed=False)
graph.vs["name"] = [f"Seq {i}" for i in range(len(dist_matrix))]

# Set edge attributes
graph.es["width"] = [5 * weight for weight in weights]  # Scale width for visualization

# Circular layout
layout = graph.layout("circle")

# Visualization with edge widths based on distance
igraph.plot(
    graph,
    "results/t1_circular_similarity_network.png",
    layout=layout,
    bbox=(1920, 1440),  # Increased size
    margin=40,  # Increased margin
    vertex_label=[name[-3:] for name in graph.vs["name"]],
    vertex_size=20,  # Increased vertex size
    vertex_color="red",
    edge_width=graph.es["width"],
    edge_color="gray",
)
# Calculate network topology metrics
diameter = graph.diameter()
girth = graph.girth() if graph.girth() != float("inf") else "Infinity"
radius = graph.radius()
avg_path_length = graph.average_path_length()
density = graph.density()
assortativity = graph.assortativity_degree()
clustering_coeff_avg = graph.transitivity_avglocal_undirected()
clustering_coeff_global = graph.transitivity_undirected()

# Degree statistics
degrees = graph.degree()
avg_degree = np.mean(degrees)
degree_std = np.std(degrees)

# Prepare metrics for the table
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
print("Network topology metrics:" + "\n")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# Create a DataFrame for the metrics
import pandas as pd

metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

# Plot the metrics table
import matplotlib.pyplot as plt

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
plt.savefig("results/network_topology_metrics.png")
plt.close()
