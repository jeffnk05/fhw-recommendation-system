# Einfacher Recommender basierend auf deinem Notebook

import pandas as pd
from lenskit.basic import topn
from lenskit.knn import ItemKNNScorer
from lenskit.batch import recommend
from lenskit.data import from_interactions_df
from lenskit.pipeline import topn_pipeline

def setup_recommender():
    """Setup Funktion für den Recommender"""
    # 1. Daten laden 
    df = pd.read_csv('../ALS/data/ratings.csv')
    books_df = pd.read_csv('../ALS/data/books.csv')
    dataset = from_interactions_df(df, user_col='user_id', item_col='book_id', rating_col='rating')

    # 2. Bestes Modell aus deinen Tests verwenden
    model_ii = ItemKNNScorer(
        max_nbrs=50,
        min_nbrs=3,
        min_sim=0.15
    )

    # 3. Pipeline erstellen und trainieren
    pipe_ii = topn_pipeline(model_ii)
    pipe_ii.train(dataset)
    
    return pipe_ii, books_df

def get_recommendations_for_user(pipe_ii, books_df, user_id, n_recommendations=10):
    """
    Gibt Buchempfehlungen für einen User zurück
    """
    # Empfehlungen generieren
    recs = recommend(pipe_ii, [user_id], n_recommendations)
    
    if not recs:
        print(f"Keine Empfehlungen für User {user_id} gefunden")
        return
    
    # Durch die Empfehlungen iterieren
    for user_key, item_list in recs:
        print(f"\n=== EMPFEHLUNGEN FÜR USER {user_key.user_id} ===")
        
        # ItemList zu DataFrame konvertieren
        recommendations_df = item_list.to_df()
        
        # Mit Buchtiteln anreichern - verwende 'item_id'
        recommendations_with_titles = recommendations_df.merge(
            books_df[['book_id', 'title']], 
            left_on='item_id', 
            right_on='book_id', 
            how='left'
        )
        
        # Empfehlungen ausgeben
        for i, (idx, row) in enumerate(recommendations_with_titles.head(n_recommendations).iterrows()):
            title = row['title'] if pd.notna(row['title']) else f"Book ID {row['item_id']}"
            score = row['score'] if 'score' in row else 'N/A'
            print(f"{i+1}. {title} (Score: {score})") 
            

if __name__ == '__main__':
    # Setup
    pipe_ii, books_df = setup_recommender()
    
    print("=== BUCHEMPFEHLUNGEN ===")
    
    # Empfehlungen für User 234
    get_recommendations_for_user(pipe_ii, books_df, 234, 10)
    
    # Weitere User testen
    print("\n" + "="*60)
    get_recommendations_for_user(pipe_ii, books_df, 123, 5)
    
    print("\n" + "="*60)
    get_recommendations_for_user(pipe_ii, books_df, 456, 8)