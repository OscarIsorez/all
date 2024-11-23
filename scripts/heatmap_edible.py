import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data loading (replace with your actual dataset)
df = pd.read_pickle('processed/mushroom.pkl.gz')

# Define two attributes that separate well, e.g., 'odor' and 'spore_print_color'
attribute1 = 'odor'
attribute2 = 'spore_print_color'

# Create a contingency table for the two attributes with respect to 'kind'
contingency_table = pd.crosstab([df[attribute1], df[attribute2]], df['kind'])

# Plot a heatmap of the contingency table
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm")
plt.title(f'Heatmap of {attribute1} and {attribute2} by Edibility')
plt.xlabel(attribute2)
plt.ylabel(attribute1)
plt.show()

