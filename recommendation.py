import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon.csv")

df = df[['product_id', 'product_name', 'rating']]
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df = df.dropna()
df = df.drop_duplicates(subset='product_name')
df = df.reset_index(drop=True)

print("Dataset Loaded Successfully ✅")
print(df.head())


# ---------------- BUILD MODEL ----------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['product_name'])

similarity = cosine_similarity(tfidf_matrix)


# ---------------- RECOMMEND FUNCTION ----------------
def recommend(product_name, top_n=5):

    product_name = product_name.lower()

    matches = df[df['product_name'].str.lower().str.contains(product_name)]

    if matches.empty:
        print("\nProduct not found ❌")
        return

    idx = matches.index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nSelected Product:")
    print(df.iloc[idx]['product_name'])

    print("\nRecommended Products:\n")

    count = 0
    for i in scores:
        if i[0] != idx:
            print(df.iloc[i[0]]['product_name'])
            count += 1
        if count >= top_n:
            break


# ---------------- EVALUATION FUNCTION ----------------
def evaluate_model(top_n=5, test_size=200):

    correct = 0
    total_recommended = 0
    total_relevant = len(df[df['rating'] >= 4.0])

    test_size = min(test_size, len(df))

    for index in range(test_size):

        scores = list(enumerate(similarity[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommended_indices = [i[0] for i in scores[1:top_n+1]]

        for i in recommended_indices:
            if df.iloc[i]['rating'] >= 4.0:
                correct += 1

        total_recommended += top_n

    precision = correct / total_recommended if total_recommended else 0
    recall = correct / total_relevant if total_relevant else 0

    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    print("\n📊 Model Evaluation Results (Percentage):")
    print(f"Precision@{top_n}: {precision*100:.4f}")
    print(f"Recall@{top_n}: {recall*100:.4f}")
    print(f"F1-Score@{top_n}: {f1_score*100:.4f}")


# ---------------- MAIN PROGRAM ----------------
if __name__ == "__main__":

    evaluate_model()

    while True:
        user_input = input("\nEnter product name (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("Exiting Recommendation System 👋")
            break

        recommend(user_input)