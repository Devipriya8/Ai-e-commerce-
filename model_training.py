import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("amazon.csv")

# Keep only needed columns
df = df[["product_name", "user_id", "rating"]]

# Convert rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Drop missing values
df = df.dropna()

# Create pivot table
product_user_matrix = df.pivot_table(
    index="product_name",
    columns="user_id",
    values="rating",
    aggfunc="mean"
).fillna(0)

# Compute similarity
similarity = cosine_similarity(product_user_matrix)

# Save model files
pickle.dump(product_user_matrix, open("matrix.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Model trained successfully ✅")