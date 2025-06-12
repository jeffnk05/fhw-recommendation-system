import pandas as pd
import numpy as np
from lenskit.knn import ItemKNNScorer
from lenskit.batch import recommend
from lenskit.data import from_interactions_df
from lenskit.pipeline import topn_pipeline

class BookRecommender:
    def __init__(self, ratings_path, books_path):
        """Lädt Daten und trainiert Modell"""
        print("Lade Daten und trainiere Modell...")
        
        self.df = pd.read_csv(ratings_path)
        self.books_df = pd.read_csv(books_path)
        self.dataset = from_interactions_df(
            self.df, user_col='user_id', item_col='book_id', rating_col='rating'
        )
        
        model = ItemKNNScorer(max_nbrs=22, min_nbrs=4, min_sim=0.22)
        self.pipeline = topn_pipeline(model)
        self.pipeline.train(self.dataset)
        
        print("Modell trainiert!\n")
    
    def get_recommendations(self, user_id, n=10):
        """Empfehlungen für Nutzer - gibt Liste von [titel, score] zurück"""
        if user_id not in self.df['user_id'].values:
            return None
        
        # Bereits bewertete Bücher
        user_rated = set(self.df[self.df['user_id'] == user_id]['book_id'].values)
        
        try:
            # ML-Empfehlungen
            recs = recommend(self.pipeline, [user_id], n * 2)
            if recs is not None and not recs.empty:
                recs = recs[~recs['book_id'].isin(user_rated)].head(n)
                if not recs.empty:
                    result = recs.merge(self.books_df[['book_id', 'title']], on='book_id', how='left')
                    return result[['title', 'score']].values.tolist()
        except:
            pass
        
        # Fallback: Populäre Bücher
        popular = self.df.groupby('book_id').agg({'rating': ['mean', 'count']})
        popular.columns = ['avg_rating', 'num_ratings']
        popular = popular.reset_index()
        
        popular = popular[
            (popular['num_ratings'] >= 50) & 
            (popular['avg_rating'] >= 4.0) &
            (~popular['book_id'].isin(user_rated))
        ]
        popular['score'] = popular['avg_rating'] * np.log1p(popular['num_ratings'])
        popular = popular.nlargest(n, 'score')
        
        result = popular.merge(self.books_df[['book_id', 'title']], on='book_id', how='left')
        return result[['title', 'score']].values.tolist()
    
    def get_similar_books(self, book_id, n=10):
        """Ähnliche Bücher - gibt Liste von [titel, score] zurück"""
        if book_id not in self.df['book_id'].values:
            return None
        
        # Nutzer die das Buch mochten
        users_who_liked = self.df[(self.df['book_id'] == book_id) & (self.df['rating'] >= 4.0)]['user_id'].unique()
        if len(users_who_liked) == 0:
            return None
        
        # Ähnliche Bücher
        similar_books = self.df[
            (self.df['user_id'].isin(users_who_liked)) & 
            (self.df['book_id'] != book_id) & 
            (self.df['rating'] >= 4.0)
        ]
        
        book_stats = similar_books.groupby('book_id').agg({
            'rating': 'mean',
            'user_id': 'nunique'
        })
        book_stats.columns = ['avg_rating', 'unique_users']
        book_stats = book_stats.reset_index()
        
        book_stats['score'] = book_stats['avg_rating'] * np.log1p(book_stats['unique_users'])
        top_similar = book_stats.nlargest(n, 'score')
        
        result = top_similar.merge(self.books_df[['book_id', 'title']], on='book_id', how='left')
        return result[['title', 'score']].values.tolist()
    
    def print_recommendations(self, user_id, n=10):
        """Druckt Empfehlungen schön formatiert"""
        recs = self.get_recommendations(user_id, n)
        
        if recs:
            print(f"📚 Top {len(recs)} Empfehlungen für Nutzer {user_id}:")
            print("-" * 60)
            for i, (title, score) in enumerate(recs, 1):
                print(f"{i:2d}. {title:<45} (Score: {score:.2f})")
        else:
            print(f"❌ Keine Empfehlungen für Nutzer {user_id} gefunden")
        print()
    
    def print_similar_books(self, book_id, n=10):
        """Druckt ähnliche Bücher schön formatiert"""
        # Originalbuch anzeigen
        original = self.books_df[self.books_df['book_id'] == book_id]
        if not original.empty:
            print(f"📖 Original: {original.iloc[0]['title']}")
        
        similar = self.get_similar_books(book_id, n)
        
        if similar:
            print(f"🔍 Top {len(similar)} ähnliche Bücher:")
            print("-" * 60)
            for i, (title, score) in enumerate(similar, 1):
                print(f"{i:2d}. {title:<45} (Score: {score:.2f})")
        else:
            print(f"❌ Keine ähnlichen Bücher zu Buch {book_id} gefunden")
        print()

# ===== VERWENDUNG =====
if __name__ == "__main__":
    # Recommender erstellen
    recommender = BookRecommender('../ALS/data/ratings.csv', '../ALS/data/books.csv')
    
    # Demo
    print("🎯 DEMO DES EMPFEHLUNGSSYSTEMS")
    print("=" * 60)
    
    # Beispiel 1: Empfehlungen für Nutzer
    recommender.print_recommendations(user_id=50, n=5)
    
    # Beispiel 2: Ähnliche Bücher
    recommender.print_similar_books(book_id=11, n=5)
    
    # Beispiel 3: Rohe Daten (für weitere Verarbeitung)
    print("📋 Rohe Daten (für weitere Verarbeitung):")
    print("-" * 60)
    recs = recommender.get_recommendations(50, 3)
    print(f"Empfehlungen: {recs}")
    
    similar = recommender.get_similar_books(1, 3)
    print(f"Ähnliche Bücher: {similar}")
    
    print("\n" + "=" * 60)
    print("✅ USAGE:")
    print("recommender = BookRecommender('ratings.csv', 'books.csv')")
    print("recs = recommender.get_recommendations(user_id, n=10)")
    print("similar = recommender.get_similar_books(book_id, n=10)")
    print("=" * 60)