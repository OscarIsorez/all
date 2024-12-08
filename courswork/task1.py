import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path = "data/sequences.txt.xz"
sequences = []

try:
    with lzma.open(file_path, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                seq_record = SeqRecord(Seq(line), id=f"seq{i+1}")
                # print(seq_record)
                sequences.append(seq_record)
except Exception as e:
    print(f"Error reading file: {e}")

if not sequences:
    print("No sequences loaded. Please check the file path and content.")
else:
    print(f"Loaded {len(sequences)} sequences.")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"




    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            score = aligner.score(str(sequences[i].seq), str(sequences[j].seq))
            distance_matrix[i, j] = distance_matrix[j, i] = score

    print("Pairwise distance matrix computed.")

    def plot_distance_matrix(matrix, title="Pairwise Distance Matrix"):
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(matrix, cmap="viridis")
        fig.colorbar(cax)
        plt.title(title)
        plt.savefig("results/t1_distance_matrix.png")
        plt.show()

    plot_distance_matrix(distance_matrix)

    def plot_clustermap(data):
        sns.clustermap(data, cmap="viridis", figsize=(10, 10))

    plot_clustermap(distance_matrix)

    upper_triangle_indices = np.triu_indices(num_sequences, k=1)
    upper_triangle_values = distance_matrix[upper_triangle_indices]
    cutoff_threshold = np.percentile(upper_triangle_values, 90)
    print(f"Cutoff threshold (90th percentile): {cutoff_threshold}")

    edges = [
        (i, j)
        for i in range(num_sequences)
        for j in range(i + 1, num_sequences)
        if distance_matrix[i, j] <= cutoff_threshold
    ]

    graph = ig.Graph(edges=edges)
    graph.vs["name"] = [record.id for record in sequences]

    print(f"Graph created with {graph.vcount()} nodes and {graph.ecount()} edges.")

    layout = graph.layout("fr")  # Fruchterman-Reingold layout
    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=None,
        vertex_size=10,
        edge_width=0.5,
        vertex_color="blue",
        edge_color="gray",
    )
    plt.title("Network Visualization")
    plt.savefig("results/t1_network_visualization.png")
    # plt.show()

    statistics = {
        "Number of Nodes": graph.vcount(),
        "Number of Edges": graph.ecount(),
        "Average Degree": sum(graph.degree()) / graph.vcount(),
        "Clustering Coefficient": graph.transitivity_undirected(),
        "Number of Components": len(graph.components()),
    }
    print(statistics)

    stats_df = pd.DataFrame(list(statistics.items()), columns=["Property", "Value"])
    print(stats_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title("Network Topological Properties")
    # plt.show()

    clusters = graph.community_multilevel()
    graph.vs["cluster"] = clusters.membership

    num_clusters = len(set(clusters.membership))
    palette = ig.drawing.colors.ClusterColoringPalette(num_clusters)
    graph.vs["color"] = [palette[cluster] for cluster in clusters.membership]

    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=None,
        vertex_size=10,
        edge_width=0.5,
        vertex_color=graph.vs["color"],
        edge_color="gray",
    )
    plt.title("Network Visualization with Clusters")
    plt.savefig("results/t1_network_clusters.png")
    # plt.show()
