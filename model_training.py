import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load user-item matrix
matrix = pd.read_csv("user_item_matrix.csv", index_col=0)

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(matrix)

print("✅ Model training completed")

# Example recommendation
def recommend(user_id, n_recommendations=5):
    if user_id not in matrix.index:
        print("User not found")
        return

    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=n_recommendations + 1)

    similar_users = matrix.index[indices.flatten()[1:]]
    print(f"Recommendations based on similar users for {user_id}:")
    print(similar_users.tolist())

# Test
recommend(matrix.index[0])
