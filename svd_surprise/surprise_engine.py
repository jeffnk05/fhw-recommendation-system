import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# ---------------------------
# Load data
# ---------------------------
ratings = pd.read_csv("ratings.csv", dtype=str, on_bad_lines='skip')
ratings = ratings[pd.to_numeric(ratings['rating'], errors='coerce').notnull()]
ratings['rating'] = ratings['rating'].astype(float)

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# ---------------------------
# Train/test split
# ---------------------------
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ---------------------------
# Train the model
# ---------------------------
algo = SVD(n_factors=20, random_state=42)
algo.fit(trainset)

# ---------------------------
# Recommend books for a user
# ---------------------------
def recommend_for_user(user_id, algo, trainset, books_df=None, n=5):
    all_items = trainset._raw2inner_id_items.keys()
    rated_items = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]]) if user_id in trainset._raw2inner_id_users else set()
    candidates = [iid for iid in all_items if iid not in rated_items]

    predictions = [algo.predict(user_id, iid) for iid in candidates]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    results = []
    for pred in top_predictions:
        book_id = pred.iid
        if books_df is not None and book_id in books_df['book_id'].values:
            title = books_df[books_df['book_id'] == book_id]['title'].values[0]
        else:
            title = f"Book ID: {book_id}"
        results.append((title, pred.est))
    return results

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    try:
        books = pd.read_csv("books.csv", dtype=str, on_bad_lines='skip')
        books['book_id'] = books['book_id'].astype(str).str.strip()
        books['title'] = books['title'].astype(str).str.strip()
    except:
        books = None

    user_input = input("\nEnter user_id to get recommendations: ").strip()
    recommendations = recommend_for_user(user_input, algo, trainset, books_df=books, n=5)

    print(f"\nTop 5 book recommendations for user {user_input}:")
    for title, score in recommendations:
        print(f" - {title} (predicted rating: {score:.2f})")
