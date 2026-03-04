import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon.csv")

df = df[['product_id', 'product_name', 'rating']]
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna().drop_duplicates(subset='product_name')
df = df.reset_index(drop=True)

print("Dataset Loaded Successfully ✅")


# ---------------- CONTENT MODEL ----------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['product_name'])
content_similarity = cosine_similarity(tfidf_matrix)

print("Content-Based Model Ready ✅")


# ---------------- SIMULATE USER DATA ----------------
num_users = 30
data = []

for user_id in range(1, num_users + 1):
    sampled_products = df.sample(n=30, random_state=user_id)
    for _, row in sampled_products.iterrows():
        data.append([user_id, row['product_id'], row['rating']])

ratings_df = pd.DataFrame(data, columns=['user_id', 'product_id', 'rating'])

user_item_matrix = ratings_df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating'
).fillna(0)

print("Collaborative Model Ready ✅")


# ---------------- HYBRID RECOMMENDATION ----------------
def hybrid_recommend(user_id, product_name, top_n=5):

    if user_id not in user_item_matrix.index:
        print("User not found ❌")
        return

    matches = df[df['product_name'].str.lower().str.contains(product_name.lower())]
    if matches.empty:
        print("Product not found ❌")
        return

    product_index = matches.index[0]

    # ---------- Content Scores ----------
    content_scores = content_similarity[product_index]

    # ---------- Collaborative Scores ----------
    user_ratings = user_item_matrix.loc[user_id]

    collab_scores = []

    for pid in df['product_id']:
        if pid in user_ratings.index:
            collab_scores.append(user_ratings[pid])
        else:
            collab_scores.append(0)

    collab_scores = np.array(collab_scores)

    # Normalize collaborative scores
    if np.max(collab_scores) != 0:
        collab_scores = collab_scores / np.max(collab_scores)

    # ---------- Hybrid Score ----------
    hybrid_scores = 0.5 * content_scores + 0.5 * collab_scores

    top_indices = np.argsort(hybrid_scores)[::-1]

    print(f"\n🔥 Hybrid Recommendations for User {user_id}:\n")

    count = 0
    for idx in top_indices:
        if idx != product_index:
            print(df.iloc[idx]['product_name'])
            count += 1
        if count >= top_n:
            break


# ---------------- RUN ----------------
if __name__ == "__main__":

    while True:
        user_input = input("\nEnter User ID (1-30) or 'exit': ")

        if user_input.lower() == "exit":
            break

        product_input = input("Enter product name: ")

        hybrid_recommend(int(user_input), product_input)