import lzma
import pandas as pd

# Specify the path to the compressed file
file_path = "data/sequences.txt.xz"

# Open the LZMA-compressed file and read it as text
with lzma.open(file_path, mode='rt') as file:
    # Read the file into a pandas DataFrame (adjust delimiter and other options as needed)
    df = pd.read_csv(file, delimiter='\t')  # Assuming tab-separated values

# Display the DataFrame
print(df)

