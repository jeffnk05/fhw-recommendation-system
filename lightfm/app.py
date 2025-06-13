import streamlit as st
import json
import numpy as np
import pickle
import scipy.sparse
import sys
import os

st.set_page_config(page_title="📚 LightFM Book Recommender", layout="wide")
st.title("📚 Book Recommender Dashboard")
st.write("✅ Python executable:", sys.executable)

# Sidebar - Upload model assets
st.sidebar.header("📂 Upload Exported Model Files")
model_file = st.sidebar.file_uploader("Upload light_model.pkl", type="pkl")
metadata_file = st.sidebar.file_uploader("Upload model_metadata.json", type="json")
item_features_file = st.sidebar.file_uploader("Upload item_features.npz", type="npz")
book_meta_file = st.sidebar.file_uploader("Upload book_metadata.json", type="json")
interactions_file = st.sidebar.file_uploader("Upload train_interactions.npz (optional)", type="npz")

# Load assets
model, metadata, item_features, book_meta, interactions = None, {}, None, {}, None

if model_file:
    model = pickle.load(model_file)
    st.sidebar.success("✅ Model loaded")

if metadata_file:
    metadata = json.load(metadata_file)
    st.sidebar.success("✅ Metadata loaded")

if item_features_file:
    item_features = scipy.sparse.load_npz(item_features_file)
    st.sidebar.success("✅ Item features loaded")

if book_meta_file:
    book_meta = json.load(book_meta_file)
    st.sidebar.success("✅ Book metadata loaded")

if interactions_file:
    interactions = scipy.sparse.load_npz(interactions_file)
    st.sidebar.success("✅ Interactions loaded")

# Recommendation Interface
if model is not None and metadata and item_features is not None and book_meta:
    st.header("🎯 Get Recommendations")
    user_input = st.text_input("Enter User ID:")
    book_input = st.text_input("Enter Book ID (to find similar books):")

    user_mapping = metadata.get("user_mapping", {})
    item_mapping = metadata.get("item_mapping", {})
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}

    # Recommend for user
    if user_input and user_input.isdigit():
        user_id = int(user_input)
        if str(user_id) in user_mapping:
            internal_uid = user_mapping[str(user_id)]
            n_items = item_features.shape[0]
            scores = model.predict(internal_uid, np.arange(n_items), item_features=item_features)

            if interactions is not None:
                known_items = set(interactions.tocsr()[internal_uid].indices)
            else:
                known_items = set()

            ranked = [(score, i) for i, score in enumerate(scores) if i not in known_items and i in reverse_item_mapping]
            ranked.sort(reverse=True)
            top_n = ranked[:10]

            st.subheader(f"📬 Top Recommendations for User {user_id}")
            for i, (score, iid) in enumerate(top_n, 1):
                book_id = reverse_item_mapping.get(iid)
                meta = book_meta.get(str(book_id), {})
                st.markdown(f"**{i}. {meta.get('title', 'Unknown Title')}**")
                st.caption(f"Author: {meta.get('author', '?')} | Rating: {meta.get('avg_rating', '?')} | ID: {book_id} | Score: {score:.4f}")
        else:
            st.warning("⚠️ User ID not found in training data.")

    # Find similar books
    if book_input and book_input.isdigit():
        book_id = int(book_input)
        if str(book_id) in item_mapping:
            internal_bid = item_mapping[str(book_id)]
            emb = model.item_embeddings
            query_vec = emb[internal_bid].reshape(1, -1)
            sims = emb @ query_vec.T
            scores = sims.flatten()

            ranked = [(score, i) for i, score in enumerate(scores) if i != internal_bid and i in reverse_item_mapping]
            ranked.sort(reverse=True)
            top_n = ranked[:10]

            query_meta = book_meta.get(str(book_id), {})
            st.subheader(f"🔍 Books Similar to: {query_meta.get('title', '?')}")
            st.markdown(f"**📖 Target Book**")
            st.caption(f"Title: {query_meta.get('title', '?')}\n\nAuthor: {query_meta.get('author', '?')} | Rating: {query_meta.get('avg_rating', '?')} | ID: {book_id}")

            for i, (score, iid) in enumerate(top_n, 1):
                similar_id = reverse_item_mapping.get(iid)
                if similar_id is not None:
                    meta = book_meta.get(str(similar_id), {})
                    st.markdown(f"**{i}. {meta.get('title', 'Unknown')}**")
                    st.caption(f"Author: {meta.get('author', '?')} | Rating: {meta.get('avg_rating', '?')} | ID: {similar_id} | Similarity: {score:.4f}")
        else:
            st.warning("⚠️ Book ID not found in training data.")
else:
    st.info("👆 Upload all required files to begin.")