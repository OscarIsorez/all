import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load your dataset (replace with your actual data)
df = pd.read_pickle("processed/mushroom.pkl.gz")

# Step 1: Plot count plots for each attribute to visually inspect separation
attributes = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 
              'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
              'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 
              'stalk_surface_below_ring', 'stalk_color_above_ring', 
              'stalk_color_below_ring', 'veil_type', 'veil_color', 
              'ring_number', 'ring_type', 'spore_print_color', 
              'population', 'habitat']

for attribute in attributes:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=attribute, hue='kind', data=df)
    plt.title(f'Distribution of {attribute} by Edibility (kind)')
    plt.show()

# Step 2: Calculate Chi-square for each attribute against 'kind'
chi2_scores = {}
for attribute in attributes:
    contingency_table = pd.crosstab(df[attribute], df['kind'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_scores[attribute] = chi2

# Sort attributes by Chi-square score (higher means more association)
sorted_attributes = sorted(chi2_scores, key=chi2_scores.get, reverse=True)
print("Top attributes based on Chi-square test:", sorted_attributes[:5])

# Step 3: Select two attributes from the top list for visualization
# For example, if 'odor' and 'spore_print_color' are the top two:
selected_attributes = ['odor', 'spore_print_color']

# Step 4: Joint Distribution Plot
sns.jointplot(x=selected_attributes[0], y=selected_attributes[1], hue="kind", data=df, kind="kde", palette="coolwarm")
plt.suptitle(f'Joint Distribution of {selected_attributes[0]} and {selected_attributes[1]} by Edibility', y=1.02)
plt.show()

