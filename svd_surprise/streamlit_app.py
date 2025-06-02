import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
ratings = pd.read_csv("ratings.csv", dtype=str, on_bad_lines='skip')
ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
ratings['rating'] = ratings['rating'].astype(float)

try:
    books = pd.read_csv("books.csv", dtype=str, on_bad_lines='skip')
    books['book_id'] = books['book_id'].astype(str).str.strip()
    books['title'] = books['title'].astype(str).str.strip()
except:
    books = None

# Train model
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=20, random_state=42)
algo.fit(trainset)

# Get user ratings
def get_user_ratings(user_id):
    user_ratings = ratings[ratings['user_id'] == user_id]
    if books is not None:
        user_ratings = user_ratings.merge(books[['book_id', 'title']], on='book_id', how='left')
    return user_ratings[['title', 'rating']] if books is not None else user_ratings[['book_id', 'rating']]

# Recommend books
def recommend_for_user(user_id, n=5):
    all_items = trainset._raw2inner_id_items.keys()
    rated_items = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]]) if user_id in trainset._raw2inner_id_users else set()
    candidates = [iid for iid in all_items if iid not in rated_items]

    predictions = [algo.predict(user_id, iid) for iid in candidates]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    results = []
    for pred in top_predictions:
        book_id = pred.iid
        title = books[books['book_id'] == book_id]['title'].values[0] if books is not None and book_id in books['book_id'].values else f"Book ID: {book_id}"
        results.append((title, pred.est))
    return results

# ----------------------------
# Streamlit Interface
# ----------------------------
st.title("📚 Book Recommender (SVD + Surprise)")

user_ids = ratings['user_id'].unique().tolist()
selected_user = st.selectbox("Select a user", user_ids)

if selected_user:
    st.subheader("📖 Books Rated by User")
    user_history = get_user_ratings(selected_user)
    st.dataframe(user_history.rename(columns={'title': 'Book Title', 'rating': 'Rating'}))

    st.subheader("✨ Recommended Books")
    recommendations = recommend_for_user(selected_user, n=5)
    for title, score in recommendations:
        st.markdown(f"- **{title}** (Predicted rating: {score:.2f})")
