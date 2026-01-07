import pandas as pd

matrix = pd.read_csv("user_item_matrix.csv", index_col=0)

print("Users:", matrix.shape[0])
print("Items:", matrix.shape[1])

sparsity = 1.0 - (matrix.astype(bool).sum().sum() / (matrix.shape[0] * matrix.shape[1]))
print("Matrix Sparsity:", round(sparsity * 100, 2), "%")

print("✅ Evaluation completed")
