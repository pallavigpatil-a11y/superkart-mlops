
import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

SOURCE_FILE = "data/SuperKart.csv"

print("Loading:", SOURCE_FILE)

df = pd.read_csv(SOURCE_FILE)

print("Original dataset shape:", df.shape)

# Normalize sugar-content category
df["Product_Sugar_Content"] = (
    df["Product_Sugar_Content"]
    .replace({"reg": "Regular"})
)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Remove identifier
if "Product_Id" in df.columns:
    df = df.drop(columns=["Product_Id"])

# Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
df.to_csv("data/superkart_cleaned.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("✅ Data preparation completed")
