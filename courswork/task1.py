import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd

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
else:
    print(f"Loaded {len(sequences)} sequences.")

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"

    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            score = aligner.score(str(sequences[i].seq), str(sequences[j].seq))
            distance_matrix[i, j] = distance_matrix[j, i] = -score

    threshold = 0.5 * distance_matrix.max()
    edges = [
        (i, j)
        for i in range(num_sequences)
        for j in range(i + 1, num_sequences)
        if distance_matrix[i, j] <= threshold
    ]

    # Create the graph
    graph = ig.Graph(edges=edges)
    graph.vs["name"] = [record.id for record in sequences]

    print(f"Graph created with {graph.vcount()} nodes and {graph.ecount()} edges.")

    layout = graph.layout("fr")
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
    plt.show()

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
    plt.show()

    clusters = graph.community_multilevel()
    graph.vs["cluster"] = clusters.membership

    palette = ig.drawing.colors.ClusterColoringPalette(len(clusters))
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
    plt.show()
