import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# ----------------------------
# Load data
# ----------------------------
ratings = pd.read_csv("ratings.csv", dtype=str, on_bad_lines='skip')
ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
ratings['rating'] = ratings['rating'].astype(float)

books = pd.read_csv("books.csv", dtype=str, on_bad_lines='skip')
books['book_id'] = books['book_id'].astype(str).str.strip()
books['title'] = books['original_title'].fillna('').astype(str).str.strip()
books['title_norm'] = books['title'].str.lower().str.strip()

# ----------------------------------
# Page Navigation
# ----------------------------------
page = st.sidebar.selectbox("Select a page", ["User-based Recommender", "Book-based Similarity"])

# ----------------------------
# USER-BASED RECOMMENDER
# ----------------------------
if page == "User-based Recommender":
    st.title("📚 Book Recommender (SVD + Surprise)")

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=20, random_state=42)
    algo.fit(trainset)

    user_ids = ratings['user_id'].unique().tolist()
    selected_user = st.selectbox("Select a user", user_ids)

    def get_user_ratings(user_id):
        user_ratings = ratings[ratings['user_id'] == user_id]
        user_ratings = user_ratings.merge(books[['book_id', 'title']], on='book_id', how='left')
        return user_ratings[['title', 'rating']]

    def recommend_for_user(user_id, n=5):
        all_items = trainset._raw2inner_id_items.keys()
        rated_items = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]]) if user_id in trainset._raw2inner_id_users else set()
        candidates = [iid for iid in all_items if iid not in rated_items]

        predictions = [algo.predict(user_id, iid) for iid in candidates]
        top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

        results = []
        for pred in top_predictions:
            book_id = pred.iid
            title = books[books['book_id'] == book_id]['title'].values[0] if book_id in books['book_id'].values else f"Book ID: {book_id}"
            results.append((title, pred.est))
        return results

    if selected_user:
        st.subheader("📖 Books Rated by User")
        st.dataframe(get_user_ratings(selected_user).rename(columns={'title': 'Book Title', 'rating': 'Rating'}))

        st.subheader("✨ Recommended Books")
        recommendations = recommend_for_user(selected_user, n=5)
        for title, score in recommendations:
            st.markdown(f"- **{title}** (Predicted rating: {score:.2f})")

# ----------------------------
# BOOK-BASED SIMILARITY
# ----------------------------
elif page == "Book-based Similarity":
    st.title("🔍 Similar Book Recommender (SVD + Scikit-learn)")

    ratings['rating'] = ratings['rating'].astype(int)
    active_users = ratings['user_id'].value_counts()
    ratings_filtered = ratings[ratings['user_id'].isin(active_users[active_users >= 5].index)]
    popular_books = ratings_filtered['book_id'].value_counts()
    ratings_filtered = ratings_filtered[ratings_filtered['book_id'].isin(popular_books[popular_books >= 5].index)]

    user_item_matrix = ratings_filtered.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
    n_components = 20
    svd = TruncatedSVD(n_components=n_components)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    book_vectors = svd.components_.T

    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(user_item_matrix.columns)}
    idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}
    isbn_to_title = books.set_index('book_id')['title'].to_dict()
    title_to_isbn = books.drop_duplicates('title_norm').set_index('title_norm')['book_id'].to_dict()

    def recommend_similar_books_by_titles(input_titles, top_n=5):
        normalized_titles = [title.lower().strip() for title in input_titles]
        valid_isbns = [
            title_to_isbn[title]
            for title in normalized_titles
            if title in title_to_isbn and title_to_isbn[title] in isbn_to_idx
        ]

        if not valid_isbns:
            return []

        input_vectors = [book_vectors[isbn_to_idx[isbn]] for isbn in valid_isbns]
        combined_vector = np.mean(input_vectors, axis=0)
        similarities = np.dot(book_vectors, combined_vector)

        for idx in [isbn_to_idx[isbn] for isbn in valid_isbns]:
            similarities[idx] = -np.inf

        similar_indices = np.argsort(similarities)[::-1][:top_n]
        similar_isbns = [idx_to_isbn[idx] for idx in similar_indices]
        similar_titles = [isbn_to_title.get(isbn, f"Unknown book_id {isbn}") for isbn in similar_isbns]

        return similar_titles

    book_titles = books['title'].dropna().unique().tolist()
    selected_books = st.multiselect("Select one or more books you like", sorted(book_titles))

    if selected_books:
        st.subheader("📚 Recommended Similar Books")
        recommendations = recommend_similar_books_by_titles(selected_books, top_n=5)

        if recommendations:
            for title in recommendations:
                st.markdown(f"- **{title}**")
        else:
            st.warning("No recommendations found for the selected titles.")
