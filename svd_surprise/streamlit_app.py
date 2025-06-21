import streamlit as st
import pandas as pd
import numpy as np
import os
import html
import base64
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

st.set_page_config(layout="wide", page_title="Buch-Recommender", page_icon="📚")

# ----------------------------
# Load data with caching
# ----------------------------
@st.cache_data
def load_books_and_ratings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'ALS', 'data'))
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    books_path = os.path.join(data_dir, 'books.csv')

    if not os.path.exists(ratings_path):
        st.error(f"Datei nicht gefunden: {ratings_path}")
        st.stop()
    if not os.path.exists(books_path):
        st.error(f"Datei nicht gefunden: {books_path}")
        st.stop()

    ratings = pd.read_csv(ratings_path, dtype=str, on_bad_lines='skip')
    ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
    ratings['rating'] = ratings['rating'].astype(float)

    books = pd.read_csv(books_path, dtype=str, on_bad_lines='skip')
    books['book_id'] = books['book_id'].astype(str).str.strip()
    books['title'] = books['original_title'].fillna('').astype(str).str.strip()
    books['title_norm'] = books['title'].str.lower().str.strip()
    books['isbn'] = books['isbn'].astype(str).str.strip()

    return books, ratings

@st.cache_resource
def train_svd_model(ratings):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=20, random_state=42)
    algo.fit(trainset)
    return algo, trainset

@st.cache_resource
def compute_book_similarity_model(ratings_filtered):
    user_item_matrix = ratings_filtered.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=20)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    return svd, user_item_matrix

# ----------------------------------
# Page Navigation
# ----------------------------------
with open("./svd_surprise/hintergrund_frontend.png", "rb") as f:
    bg_image = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.75);
        border-radius: 12px;
        padding: 2rem 3rem;
    }}
    .sidebar-content > *:nth-child(4) {{
        margin-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1rem;
    }}
    .recommendation-list {{
        margin-top: 1.5rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }}
    .recommendation-item {{
        font-size: 1.2rem;
        margin-bottom: 0.6rem;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/29/29302.png", width=80)
st.sidebar.title("📌 Navigation")
if st.sidebar.button("🔁 App neu laden"):
    st.rerun()

page = st.sidebar.radio("Seite wählen:", ["📖 Nutzerbasierte Empfehlungen", "🔍 Ähnliche Bücher"])
books, ratings = load_books_and_ratings()

with st.sidebar:
    st.markdown("### 📊 Datenübersicht")
    st.write(f"📚 Bücher im Datensatz: {len(books):,}")
    st.write(f"🧑‍🤝‍🧑 Nutzer: {ratings['user_id'].nunique():,}")
    st.write(f"⭐ Bewertungen: {len(ratings):,}")

# ----------------------------
# USER-BASED RECOMMENDER
# ----------------------------
if page == "📖 Nutzerbasierte Empfehlungen":
    st.title("📚 Buch-Empfehlungen für Nutzer")
    st.markdown("Erhalte personalisierte Buchvorschläge basierend auf deinen bisherigen Bewertungen.")

    if 'svd_model' not in st.session_state or 'svd_trainset' not in st.session_state:
        with st.spinner("Trainiere Recommender-Modell..."):
            algo, trainset = train_svd_model(ratings)
            st.session_state.svd_model = algo
            st.session_state.svd_trainset = trainset
    else:
        algo = st.session_state.svd_model
        trainset = st.session_state.svd_trainset

    user_ids = ratings['user_id'].unique().tolist()
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox("👤 Wähle einen Benutzer:", user_ids)
    with col2:
        n_recs = st.slider("🔢 Anzahl Empfehlungen:", 1, 10, 5)

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
            book_id_str = str(pred.iid).strip()
            title_row = books[books['book_id'] == book_id_str]
            if not title_row.empty and pd.notna(title_row['title'].values[0]):
                title = title_row['title'].values[0]
            else:
                title = f"Book ID: {book_id_str}"
            title = html.escape(title)
            results.append((title, pred.est))
        return results

    if selected_user:
        with st.expander("📖 Bücher, die der Nutzer bereits bewertet hat", expanded=False):
            st.dataframe(get_user_ratings(selected_user).rename(columns={'title': 'Buchtitel', 'rating': 'Bewertung'}))

        with st.spinner("Berechne Empfehlungen..."):
            recommendations = recommend_for_user(selected_user, n=n_recs)

        st.markdown("### ✨ Top-Empfehlungen:")
        with st.container():
            st.markdown("<div class='recommendation-list'>", unsafe_allow_html=True)
            for i, (title, score) in enumerate(recommendations, 1):
                st.markdown(f"<div class='recommendation-item'><b>{i}. {title}</b> — ⭐ {score:.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# BOOK-BASED SIMILARITY
# ----------------------------
elif page == "🔍 Ähnliche Bücher":
    st.title("🔍 Ähnliche Bücher finden")
    st.markdown("Wähle ein oder mehrere Bücher und erhalte inhaltlich ähnliche Empfehlungen.")

    ratings['rating'] = ratings['rating'].astype(int)
    active_users = ratings['user_id'].value_counts()
    ratings_filtered = ratings[ratings['user_id'].isin(active_users[active_users >= 5].index)]
    popular_books = ratings_filtered['book_id'].value_counts()
    ratings_filtered = ratings_filtered[ratings_filtered['book_id'].isin(popular_books[popular_books >= 5].index)]

    svd, user_item_matrix = compute_book_similarity_model(ratings_filtered)
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
        similar_titles = [isbn_to_title.get(isbn, f"Unbekanntes Buch-ID {isbn}") for isbn in similar_isbns]

        return similar_titles

    book_titles = books['title'].dropna().unique().tolist()
    selected_books = st.multiselect(
        "📚 Wähle ein oder mehrere Bücher:",
        sorted(book_titles),
        help="Wähle Titel aus, um ähnliche Bücher zu finden"
    )

    if selected_books:
        with st.spinner("Berechne ähnliche Bücher..."):
            recommendations = recommend_similar_books_by_titles(selected_books, top_n=5)

        st.markdown("### 📘 Ähnliche Buchempfehlungen:")
        if recommendations:
            for title in recommendations:
                st.markdown(f"- **{title}**")
        else:
            st.warning("Keine Empfehlungen gefunden für die ausgewählten Titel.")
