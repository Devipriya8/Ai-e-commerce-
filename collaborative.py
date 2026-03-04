import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon.csv")

df = df[['product_id', 'product_name', 'rating']]
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna().drop_duplicates(subset='product_name')
df = df.reset_index(drop=True)

print("Dataset Loaded Successfully ✅")


# ---------------- SIMULATE USER DATA ----------------
# Create fake users (since dataset doesn't have proper users)

num_users = 50

data = []

for user_id in range(1, num_users + 1):
    sampled_products = df.sample(n=20, random_state=user_id)
    for _, row in sampled_products.iterrows():
        data.append([user_id, row['product_id'], row['rating']])

ratings_df = pd.DataFrame(data, columns=['user_id', 'product_id', 'rating'])

print("User-Item Data Created ✅")


# ---------------- CREATE USER-ITEM MATRIX ----------------
user_item_matrix = ratings_df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating'
).fillna(0)

print("User-Item Matrix Built ✅")


# ---------------- COMPUTE USER SIMILARITY ----------------
user_similarity = cosine_similarity(user_item_matrix)

print("User Similarity Matrix Created ✅")


# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_for_user(user_id, top_n=5):

    if user_id not in user_item_matrix.index:
        print("User not found ❌")
        return

    user_index = user_id - 1

    similarity_scores = list(enumerate(user_similarity[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    similar_users = [i[0] for i in similarity_scores[:5]]

    recommended_products = set()

    for sim_user in similar_users:
        sim_user_id = sim_user + 1
        products = user_item_matrix.loc[sim_user_id]
        top_products = products[products > 0].index
        recommended_products.update(top_products)

    print(f"\nRecommended Products for User {user_id}:\n")

    count = 0
    for product_id in recommended_products:
        product_name = df[df['product_id'] == product_id]['product_name'].values
        if len(product_name) > 0:
            print(product_name[0])
            count += 1
        if count >= top_n:
            break


# ---------------- RUN ----------------
if __name__ == "__main__":

    while True:
        user_input = input("\nEnter User ID (1-50) or 'exit': ")

        if user_input.lower() == "exit":
            break

        recommend_for_user(int(user_input))