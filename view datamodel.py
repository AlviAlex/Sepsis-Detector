import pandas as pd
import os
import json

# Paths to both training folders
folders = [
    r"C:\Users\Alvin\Downloads\training_setA\training",
    r"C:\Users\Alvin\Downloads\training_setB\training_setB"
]

# Load all .psv files
all_data = []
for folder_path in folders:
    if os.path.exists(folder_path):
        psv_files = [f for f in os.listdir(folder_path) if f.endswith('.psv')]
        for file in psv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, sep='|')
            all_data.append(df)

# Combine into a single DataFrame
full_df = pd.concat(all_data, ignore_index=True)

# Define features used in the model
features = [col for col in full_df.columns if col not in ["SepsisLabel", "Patient_ID"]]

# Calculate the mean for each feature
feature_means = full_df[features].mean().to_dict()

# Save the means to a JSON file
with open('feature_means.json', 'w') as f:
    json.dump(feature_means, f, indent=4)

print("✅ Successfully calculated and saved feature means to 'feature_means.json'")