

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_pickle("processed/mushroom.pkl.gz")
attributes = df.columns.drop('kind')  # Exclude the target column for the loop
for attribute in attributes:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=attribute, hue='kind', data=df)
    plt.title(f'Distribution of {attribute} by Class')
    plt.show()

# Step 3: Choose two attributes that show good separation visually.


