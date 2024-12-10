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


# Seuil pour la similarité
threshold = 0.5

# Créer une liste d'arêtes à partir de la matrice de distances
edges = []
for i in range(len(dist_matrix)):
    for j in range(i + 1, len(dist_matrix)):  # Triangle supérieur
        if dist_matrix[i, j] <= threshold:
            edges.append((i, j))

# Créer le graphe avec igraph
graph = igraph.Graph(edges=edges, directed=False)
graph.vs["name"] = [f"Seq {i}" for i in range(len(dist_matrix))]

# Analyse de la topologie du réseau
print("Diamètre du graphe :", graph.diameter())
print("Rayon :", graph.radius())
print("Longueur moyenne des chemins :", graph.average_path_length())
print("Densité :", graph.density())
print("Distribution des degrés :", graph.degree_distribution())
print("Coefficient de clustering moyen :", graph.transitivity_avglocal_undirected())
print("Coefficient de clustering global :", graph.transitivity_undirected())

# Détection de communautés
clusters = graph.community_infomap()
print("Communautés détectées :", clusters.membership)

# Ajout de couleurs aux communautés
palette = sns.color_palette("husl", len(clusters))
colours = [palette[clusters.membership[v]] for v in graph.vs.indices]
graph.vs["color"] = [tuple(int(255 * c) for c in color) for color in colours]

# Visualisation du graphe
layout = graph.layout("fr")  # Force-directed layout
igraph.plot(
    graph, "sequence_similarity_network.png", layout=layout, bbox=(1280, 960), margin=50
)
