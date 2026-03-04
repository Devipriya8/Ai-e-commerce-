from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("amazon.csv")

df = df[["product_name", "user_id", "rating", "rating_count"]]
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df = df.dropna()

# Create matrix
matrix = df.pivot_table(
    index="product_name",
    columns="user_id",
    values="rating",
    aggfunc="mean"
).fillna(0)

similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(product_name):
    if product_name not in matrix.index:
        return []

    index = matrix.index.get_loc(product_name)
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    return [matrix.index[i[0]] for i in sim_scores]

# Home page
@app.route("/")
def home():
    products = matrix.index[:50]
    return render_template("index.html", products=products)

# Product page
@app.route("/product/<name>")
def product(name):
    recs = recommend(name)
    return render_template("product.html", product=name, recommendations=recs)

if __name__ == "__main__":
    app.run(port=5000, debug=True)