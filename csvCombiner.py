import os
import pandas as pd

# Directory containing the CSV files
directory = "D:/downloads"

# List to store dataframes
dataframes = []

# Iterate through all CSV files in sorted order
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".csv") and filename != "submission.csv":
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)  # Read with headers
        dataframes.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Add a numeric index column (starting from 0)
combined_df.insert(0, '', range(len(combined_df)))  # Empty string keeps no header name

# Save to submission.csv
output_file = os.path.join(directory, "submission.csv")
combined_df.to_csv(output_file, index=False)

print(f"Final submission saved to {output_file}")
