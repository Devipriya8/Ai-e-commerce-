import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("amazon.csv")

# Keep required columns
df = df[["product_name", "user_id", "rating"]]

# Convert rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna()

# Create product-user matrix
matrix = df.pivot_table(
    index="product_name",
    columns="user_id",
    values="rating",
    aggfunc="mean"
).fillna(0)

# Compute similarity
similarity = cosine_similarity(matrix)

def recommend(product_name, top_n=5):
    if product_name not in matrix.index:
        return "Product not found"

    index = matrix.index.get_loc(product_name)
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommendations = [matrix.index[i[0]] for i in sim_scores]
    return recommendations


# Example test
print("Example Recommendation:")
print(recommend(matrix.index[0]))