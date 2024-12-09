import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd

# Extract the dataset
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

    # Initialize the aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"

    # Compute the pairwise distance matrix
    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            score = aligner.score(str(sequences[i].seq), str(sequences[j].seq))
            distance_matrix[i, j] = distance_matrix[j, i] = score

    print("Pairwise distance matrix computed.")

    # Create a DataFrame from the distance matrix
    distance_df = pd.DataFrame(
        distance_matrix,
        index=[f"seq{i+1}" for i in range(num_sequences)],
        columns=[f"seq{i+1}" for i in range(num_sequences)],
    )

    # Create a clustermap plot
    sns.clustermap(distance_df, cmap="viridis", figsize=(10, 10))
    plt.title("Clustermap of Sequence Similarities")
    plt.savefig("results/t1_clustermap.png")
    plt.show()

    # Calculate the cutoff threshold based on the 80th percentile
    upper_triangle_indices = np.triu_indices(num_sequences, k=1)
    upper_triangle_values = distance_matrix[upper_triangle_indices]
    cutoff_threshold = np.percentile(upper_triangle_values, 80)
    print(f"Cutoff threshold (80th percentile): {cutoff_threshold}")

    # Define edges based on the cutoff threshold
    edges = [
        (i, j)
        for i in range(num_sequences)
        for j in range(i + 1, num_sequences)
        if distance_matrix[i, j] <= cutoff_threshold
    ]

    # Create the graph
    graph = ig.Graph(edges=edges)
    graph.vs["name"] = [record.id for record in sequences]

    print(f"Graph created with {graph.vcount()} nodes and {graph.ecount()} edges.")

    # Visualize the network
    layout = graph.layout("fr")  # Fruchterman-Reingold layout
    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=None,
        vertex_size=20,  # Adjusted vertex size
        edge_width=0.5,
        vertex_color="blue",
        edge_color="gray",
    )
    plt.title("Network Visualization")
    plt.savefig("results/t1_network_visualization.png")
    plt.show()

    # Compute network statistics
    statistics = {
        "Number of Nodes": graph.vcount(),
        "Number of Edges": graph.ecount(),
        "Average Degree": sum(graph.degree()) / graph.vcount(),
        "Clustering Coefficient": graph.transitivity_undirected(),
        "Number of Components": len(graph.components()),
    }
    print(statistics)

    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(list(statistics.items()), columns=["Property", "Value"])
    print(stats_df)

    # Display the statistics as a table
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

    # Detect communities
    clusters = graph.community_multilevel()
    graph.vs["cluster"] = clusters.membership

    # Print cluster information
    num_clusters = len(set(clusters.membership))
    cluster_sizes = [clusters.membership.count(i) for i in range(num_clusters)]
    print(f"Number of clusters: {num_clusters}")
    print(f"Cluster sizes: {cluster_sizes}")

    # Color the nodes based on cluster membership
    palette = ig.drawing.colors.ClusterColoringPalette(num_clusters)
    graph.vs["color"] = [palette[cluster] for cluster in clusters.membership]

    # Visualize the network with clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=None,
        vertex_size=20,  # Adjusted vertex size
        edge_width=0.5,
        vertex_color=graph.vs["color"],
        edge_color="gray",
    )
    plt.title("Network Visualization with Clusters")
    plt.savefig("results/t1_network_visualization_with_clusters.png")
    plt.show()

    # Plot cluster sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        range(num_clusters),
        cluster_sizes,
        color=[palette[i] for i in range(num_clusters)],
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    ax.set_title("Cluster Sizes")
    plt.savefig("results/t1_cluster_sizes.png")
    plt.show()
