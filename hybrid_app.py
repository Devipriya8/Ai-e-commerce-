import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Priya Store", layout="wide")

# ---------------- SESSION STATE ----------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# ---------------- STYLE ----------------
st.markdown("""
<style>
.navbar {
    background-color:#ff3f6c;
    padding:15px;
    font-size:26px;
    color:white;
    font-weight:bold;
}
.card {
    background-color:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar">🛍️ Priya Fashion Store</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon.csv")
df = df[['product_id','product_name','rating']]
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna().drop_duplicates(subset='product_name').reset_index(drop=True)

# ---------------- AUTO CATEGORY ----------------
def assign_category(name):
    name = name.lower()
    if any(word in name for word in ["cable","usb","charger","adapter"]):
        return "Accessories"
    elif any(word in name for word in ["tv","phone","watch","laptop"]):
        return "Electronics"
    else:
        return "Others"

df["category"] = df["product_name"].apply(assign_category)

# Simulated price
df["price"] = [random.randint(499, 4999) for _ in range(len(df))]

# ---------------- CONTENT MODEL ----------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['product_name'])
similarity = cosine_similarity(tfidf_matrix)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📦 Category")
selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df["category"].unique())
)

st.sidebar.header("🔎 Search")
search_product = st.sidebar.text_input("Search Product")

# Filter by category
if selected_category != "All":
    filtered_df = df[df["category"] == selected_category]
else:
    filtered_df = df

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(product_name, top_n=6):
    matches = filtered_df[
        filtered_df['product_name'].str.lower().str.contains(product_name.lower())
    ]
    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return [df.iloc[i[0]] for i in scores]

# ---------------- TRENDING SECTION ----------------
st.subheader(f"🔥 Trending in {selected_category}")

trending = filtered_df.sort_values(by="rating", ascending=False).head(6)
cols = st.columns(3)

for i, row in trending.iterrows():
    with cols[i % 3]:
        image_url = f"https://source.unsplash.com/400x300/?product,{row['product_name'].split()[0]}"
        st.image(image_url, use_container_width=True)
        st.write(f"**{row['product_name']}**")
        st.write(f"⭐ {row['rating']}")
        st.write(f"₹{row['price']}")

        if st.button(f"Add to Cart {row['product_id']}"):
            st.session_state.cart.append(row.to_dict())
            st.success("Added to Cart!")

# ---------------- RECOMMEND SECTION ----------------
if st.sidebar.button("Recommend"):

    results = recommend(search_product)

    if results:
        st.subheader("✨ Recommended For You")
        cols = st.columns(3)

        for i, row in enumerate(results):
            with cols[i % 3]:
                image_url = f"https://source.unsplash.com/400x300/?product,{row['product_name'].split()[0]}"
                st.image(image_url, use_container_width=True)
                st.write(f"**{row['product_name']}**")
                st.write(f"⭐ {row['rating']}")
                st.write(f"₹{row['price']}")

                if st.button(f"Add Rec {row['product_id']}"):
                    st.session_state.cart.append(row.to_dict())
                    st.success("Added to Cart!")

    else:
        st.error("Product not found!")

# ---------------- CART SECTION ----------------
st.sidebar.header("🛒 Cart")

total = 0

if st.session_state.cart:
    for i, item in enumerate(st.session_state.cart):
        st.sidebar.write(item['product_name'])
        st.sidebar.write(f"₹{item['price']}")
        total += item['price']

        if st.sidebar.button(f"Remove {i}"):
            st.session_state.cart.pop(i)
            st.rerun()

    st.sidebar.write("---")
    st.sidebar.write(f"### Total: ₹{total}")
else:
    st.sidebar.write("Cart is empty")