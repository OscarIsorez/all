import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_pickle("processed/mushroom.pkl.gz")
selected_attributes = ['odor', 'spore_print_color']

# Step 4: Joint Distribution Plot
sns.jointplot(x=selected_attributes[0], y=selected_attributes[1], hue="kind", data=df, kind="kde", palette="coolwarm")
plt.suptitle(f'Joint Distribution of {selected_attributes[0]} and {selected_attributes[1]} by Edibility', y=1.02)
plt.show()
