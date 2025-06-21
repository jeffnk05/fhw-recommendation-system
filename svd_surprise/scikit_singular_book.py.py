# recommendation_engine.py

# 1. Imports
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

# ----------------------------
# Prepare data Load
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.join(current_dir, "..", "ALS", "data")        
ratings_path = os.path.join(data_dir, "ratings.csv")
books_path = os.path.join(data_dir, "books.csv")

print("Lade ratings von:", ratings_path)
print("Lade books von:", books_path)

# 2. Load and Preprocess Data
books = pd.read_csv(books_path, dtype=str, on_bad_lines='skip')
ratings = pd.read_csv(ratings_path, dtype=str, on_bad_lines='skip')

# Spalten bereinigen
books.columns = books.columns.str.strip().str.lower()
ratings.columns = ratings.columns.str.strip().str.lower()

# Erwartete Spaltennamen prüfen
required_book_cols = {'book_id', 'title'}
required_rating_cols = {'user_id', 'book_id', 'rating'}

if not required_book_cols.issubset(books.columns):
    raise KeyError(f"Fehlende Spalten in books.csv: {required_book_cols - set(books.columns)}")
if not required_rating_cols.issubset(ratings.columns):
    raise KeyError(f"Fehlende Spalten in ratings.csv: {required_rating_cols - set(ratings.columns)}")

# Spalten säubern
books['book_id'] = books['book_id'].astype(str).str.strip()
ratings['book_id'] = ratings['book_id'].astype(str).str.strip()

ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
ratings['rating'] = ratings['rating'].astype(int)

# 3. Filter data to reduce matrix size
active_users = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(active_users[active_users >= 5].index)]

popular_books = ratings['book_id'].value_counts()
ratings = ratings[ratings['book_id'].isin(popular_books[popular_books >= 5].index)]

# 4. Create User-Item Matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)

# 5. Apply SVD
n_components = 20
svd = TruncatedSVD(n_components=n_components)
matrix_reduced = svd.fit_transform(user_item_matrix)
book_vectors = svd.components_.T

isbn_to_idx = {isbn: idx for idx, isbn in enumerate(user_item_matrix.columns)}
idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}

# 6. Recommendation Function
def recommend_similar_books(input_book_id, top_n=5):
    if input_book_id not in isbn_to_idx:
        return f"❌ book_id {input_book_id} nicht im Datensatz gefunden."
    input_vector = book_vectors[isbn_to_idx[input_book_id]]
    similarities = np.dot(book_vectors, input_vector)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return [idx_to_isbn[idx] for idx in similar_indices]

# 7. Example Usage
if __name__ == "__main__":
    example_book_id = "9296"  # Anpassen an existierende book_id in deinem Datensatz
    recommendations = recommend_similar_books(example_book_id)

    # Ausgabe korrekt behandeln, je nach Rückgabewert
    if isinstance(recommendations, list):
        print(f"Top-Empfehlungen für Buch-ID {example_book_id}:")
        for rec in recommendations:
            print(rec)
    else:
        # z. B. "❌ book_id ... nicht im Datensatz gefunden."
        print(recommendations)

