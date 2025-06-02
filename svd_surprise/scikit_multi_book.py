import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the data
books = pd.read_csv("books.csv", dtype=str, on_bad_lines='skip')
ratings = pd.read_csv("ratings.csv", dtype=str, on_bad_lines='skip')

# Clean and prepare fields
books['book_id'] = books['book_id'].astype(str).str.strip()
books['title'] = books['original_title'].fillna('').astype(str).str.strip()
books['title_norm'] = books['title'].str.lower().str.strip()


ratings['book_id'] = ratings['book_id'].astype(str).str.strip()
ratings['user_id'] = ratings['user_id'].astype(str).str.strip()

# Filter and convert rating values
ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
ratings['rating'] = ratings['rating'].astype(int)

# Filter active users and popular books
active_users = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(active_users[active_users >= 5].index)]

popular_books = ratings['book_id'].value_counts()
ratings = ratings[ratings['book_id'].isin(popular_books[popular_books >= 5].index)]

# User-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)

# Apply SVD
n_components = 20
svd = TruncatedSVD(n_components=n_components)
matrix_reduced = svd.fit_transform(user_item_matrix)
book_vectors = svd.components_.T

# Lookups
isbn_to_idx = {isbn: idx for idx, isbn in enumerate(user_item_matrix.columns)}
idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}
isbn_to_title = books.set_index('book_id')['title'].to_dict()
title_to_isbn = books.drop_duplicates('title_norm').set_index('title_norm')['book_id'].to_dict()

# Recommendation function
def recommend_similar_books_by_titles(input_titles, top_n=5):
    normalized_titles = [title.lower().strip() for title in input_titles]
    valid_isbns = [
        title_to_isbn[title]
        for title in normalized_titles
        if title in title_to_isbn and title_to_isbn[title] in isbn_to_idx
    ]

    if not valid_isbns:
        return "None of the provided book titles are in the dataset."

    input_vectors = [book_vectors[isbn_to_idx[isbn]] for isbn in valid_isbns]
    combined_vector = np.mean(input_vectors, axis=0)

    similarities = np.dot(book_vectors, combined_vector)

    for idx in [isbn_to_idx[isbn] for isbn in valid_isbns]:
        similarities[idx] = -np.inf

    similar_indices = np.argsort(similarities)[::-1][:top_n]
    similar_isbns = [idx_to_isbn[idx] for idx in similar_indices]
    similar_titles = [isbn_to_title.get(isbn, f"Unknown book_id {isbn}") for isbn in similar_isbns]

    return similar_titles

# Example usage
if __name__ == "__main__":
    input_titles = ["Harry Potter and the Philosopher's Stone"]
    recommendations = recommend_similar_books_by_titles(input_titles, top_n=5)

    print("Top recommendations based on your input books:")
    if isinstance(recommendations, list):
        for title in recommendations:
            print(f"- {title}")
    else:
        print(recommendations)
