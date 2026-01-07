import pandas as pd

# Load dataset
df = pd.read_csv("amazon.csv")
print("Initial data shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# REQUIRED columns:
# user_id, product_id, rating (or interaction)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing user or product
df.dropna(subset=['user_id', 'product_id'], inplace=True)

# Handle rating column
if 'rating' in df.columns:
    df['rating'] = df['rating'].fillna(df['rating'].mean())
else:
    df['rating'] = 1  # implicit feedback

# Fix data types
df['user_id'] = df['user_id'].astype(str)
df['product_id'] = df['product_id'].astype(str)
df['rating'] = df['rating'].astype(float)

# Remove invalid ratings
df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

print("Cleaned data shape:", df.shape)

# Create User-Item Interaction Matrix
user_item_matrix = df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    fill_value=0
)

# Save outputs
df.to_csv("cleaned_dataset.csv", index=False)
user_item_matrix.to_csv("user_item_matrix.csv")

print("✅ Data processing completed")
print("User-Item Matrix shape:", user_item_matrix.shape)
