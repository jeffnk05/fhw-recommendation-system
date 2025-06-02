import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

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
# Evaluate with RMSE
# ---------------------------
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

# ---------------------------
# Build Top-N predictions for evaluation
# ---------------------------
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]
    return top_n

def get_ground_truth(testset, threshold=4):
    truth = defaultdict(set)
    for uid, iid, rating in testset:
        if rating >= threshold:
            truth[uid].add(iid)
    return truth

def precision_recall_ndcg(top_n, ground_truth, N=5):
    precisions, recalls, ndcgs = [], [], []
    for uid, pred_items in top_n.items():
        pred_ids = [iid for (iid, _) in pred_items]
        relevant = ground_truth.get(uid, set())
        if not relevant:
            continue
        hits = [1 if iid in relevant else 0 for iid in pred_ids]
        precision = sum(hits) / N
        recall = sum(hits) / len(relevant)
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(hits)])
        ideal = sorted(hits, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal)])
        ndcg = dcg / idcg if idcg > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

top_n = get_top_n(predictions, n=5)
ground_truth = get_ground_truth(testset)
precision, recall, ndcg = precision_recall_ndcg(top_n, ground_truth, N=5)

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
    # Print evaluation metrics
    print("\n📊 Evaluation Metrics (Top-5 Recommendations):")
    print(f"RMSE:        {rmse:.4f}")
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5:    {recall:.4f}")
    print(f"nDCG@5:      {ndcg:.4f}")

    # Optional: Load books.csv for titles
    try:
        books = pd.read_csv("books.csv", dtype=str, on_bad_lines='skip')
        books['book_id'] = books['book_id'].astype(str).str.strip()
        books['title'] = books['title'].astype(str).str.strip()
    except:
        books = None

    # Recommend for a specific user
    user_input = input("\nEnter user_id to get recommendations: ").strip()
    recommendations = recommend_for_user(user_input, algo, trainset, books_df=books, n=5)

    print(f"\n📚 Top 5 book recommendations for user {user_input}:")
    for title, score in recommendations:
        print(f" - {title} (predicted rating: {score:.2f})")
