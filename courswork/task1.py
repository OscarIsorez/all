import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

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
    aligner.mode = "global"

    # Compute the pairwise distance matrix
    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            score = aligner.score(str(sequences[i].seq), str(sequences[j].seq))
            distance_matrix[i, j] = distance_matrix[j, i] = -score

    print("Pairwise distance matrix computed.")

    # Define a threshold for edge creation
    threshold = -5  # Example value; adjust based on data
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

    layout = graph.layout("fr")  # Fruchterman-Reingold layout
    ig.plot(graph, layout=layout, vertex_label=None, vertex_size=10, edge_width=0.5)
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

    # Detect communities
    clusters = graph.community_multilevel()
    graph.vs["cluster"] = clusters.membership

    # Color the nodes based on cluster membership
    palette = ig.drawing.colors.ClusterColoringPalette(len(clusters))
    graph.vs["color"] = [palette[cluster] for cluster in clusters.membership]

    # Visualize the network with clusters
    ig.plot(graph, layout=layout, vertex_size=10, edge_width=0.5)