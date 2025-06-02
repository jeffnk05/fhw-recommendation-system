# recommendation_engine.py

# 1. Imports
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# 2. Load and Preprocess Data
books = pd.read_csv("Books.csv", dtype=str, on_bad_lines='skip')
ratings = pd.read_csv("Ratings.csv", dtype=str, on_bad_lines='skip')

books.columns = books.columns.str.strip().str.replace(";;", "")
ratings.columns = ratings.columns.str.strip()

books['ISBN'] = books['ISBN'].astype(str).str.strip()
ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip()

ratings = ratings[pd.to_numeric(ratings['Book-Rating'], errors='coerce').notnull()]
ratings['Book-Rating'] = ratings['Book-Rating'].astype(int)

# 3. Filter data to reduce matrix size
# Keep users who rated at least 5 books
active_users = ratings['User-ID'].value_counts()
active_users = active_users[active_users >= 5].index
ratings = ratings[ratings['User-ID'].isin(active_users)]

# Keep books that received at least 5 ratings
popular_books = ratings['ISBN'].value_counts()
popular_books = popular_books[popular_books >= 5].index
ratings = ratings[ratings['ISBN'].isin(popular_books)]

# 4. Create User-Item Matrix
user_item_matrix = ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# 5. Apply SVD
n_components = 20
svd = TruncatedSVD(n_components=n_components)
matrix_reduced = svd.fit_transform(user_item_matrix)
book_vectors = svd.components_.T

isbn_to_idx = {isbn: idx for idx, isbn in enumerate(user_item_matrix.columns)}
idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}

# 6. Recommendation Function
def recommend_similar_books(input_isbn, top_n=5):
    if input_isbn not in isbn_to_idx:
        return f"ISBN {input_isbn} not found in dataset."
    input_vector = book_vectors[isbn_to_idx[input_isbn]]
    similarities = np.dot(book_vectors, input_vector)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return [idx_to_isbn[idx] for idx in similar_indices]

# 7. Example Usage
if __name__ == "__main__":
    example_isbn = "0140623035"  
    recommendations = recommend_similar_books(example_isbn)
    print(f"Top recommendations for ISBN {example_isbn}:")
    for rec in recommendations:
        print(rec)
