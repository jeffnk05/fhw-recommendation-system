# %pip install lightfm scikit-learn pandas numpy

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommendationSystem:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.books_df = None
        self.processed_ratings = None
        self.train_interactions = None
        self.item_features_matrix = None
        self.is_trained = False

    def load_and_process_data(self, books_path='books.csv', ratings_path='ratings.csv'):
        print("Processing data...")

        self.books_df = pd.read_csv(books_path)
        ratings_df = pd.read_csv(ratings_path)

        self.books_df['authors'] = self.books_df['authors'].fillna('Unknown')
        self.books_df['original_publication_year'] = self.books_df['original_publication_year'].fillna(
            self.books_df['original_publication_year'].median())
        self.books_df['language_code'] = self.books_df['language_code'].fillna('eng')
        self.books_df['average_rating'] = self.books_df['average_rating'].fillna(
            self.books_df['average_rating'].mean())
        self.books_df['original_title'] = self.books_df['original_title'].fillna(self.books_df['title'])

        book_features = self._create_book_features()
        self.processed_ratings = self._create_interaction_data(ratings_df)

        self.dataset = Dataset()
        self.dataset.fit(
            users=self.processed_ratings['user_id'].unique(),
            items=self.processed_ratings['book_id'].unique(),
            item_features=[feature for features in book_features for feature in features]
        )

        interactions, weights = self.dataset.build_interactions(
            [(row['user_id'], row['book_id'], row['confidence'])
             for _, row in self.processed_ratings[self.processed_ratings['implicit_feedback'] == 1].iterrows()]
        )

        self.item_features_matrix = self.dataset.build_item_features(
            [(book_id, features) for book_id, features in
             zip(self.books_df['book_id'], book_features)]
        )

        self.train_interactions, _ = random_train_test_split(
            interactions, test_percentage=0.2, random_state=42
        )
        train_weights, _ = random_train_test_split(
            weights, test_percentage=0.2, random_state=42
        )

        return interactions, weights, train_weights

    def _create_book_features(self):
        features = []

        self.books_df['rating_bucket'] = pd.cut(self.books_df['average_rating'],
                                                bins=[0, 3.5, 4.0, 4.5, 5.0],
                                                labels=['low_rating', 'medium_rating', 'high_rating', 'excellent_rating'])

        self.books_df['year_bucket'] = pd.cut(self.books_df['original_publication_year'],
                                              bins=[0, 1950, 1980, 2000, 2010, 2025],
                                              labels=['classic', 'mid_century', 'modern', 'recent', 'contemporary'])

        self.books_df['popularity_bucket'] = pd.cut(self.books_df['ratings_count'],
                                                    bins=[0, 100, 1000, 10000, float('inf')],
                                                    labels=['niche', 'moderate', 'popular', 'bestseller'])

        for _, row in self.books_df.iterrows():
            book_features = []

            primary_author = row['authors'].split(',')[0].strip()
            book_features.append(f"author:{primary_author.replace(' ', '_')}")
            book_features.append(f"language:{row['language_code']}")

            if pd.notna(row['rating_bucket']):
                book_features.append(f"rating:{row['rating_bucket']}")
            if pd.notna(row['year_bucket']):
                book_features.append(f"year:{row['year_bucket']}")
            if pd.notna(row['popularity_bucket']):
                book_features.append(f"popularity:{row['popularity_bucket']}")

            title_clean = row['original_title'].strip().replace(" ", "_").replace(":", "").replace(",", "")
            book_features.append(f"title:{title_clean}")

            features.append(book_features)

        return features

    def _create_interaction_data(self, ratings_df):
        ratings_df['implicit_feedback'] = (ratings_df['rating'] >= 4).astype(int)
        ratings_df['confidence'] = ratings_df['rating'] / 5.0
        return ratings_df

    def train_model(self, interactions, weights, train_weights):
        print("Training model...")

        self.model = LightFM(
            loss='warp',
            learning_rate=0.05,
            item_alpha=1e-6,
            user_alpha=1e-6,
            random_state=42,
            no_components=50
        )

        self.model.fit(
            interactions=self.train_interactions,
            sample_weight=train_weights,
            item_features=self.item_features_matrix,
            epochs=30,
            num_threads=4,
            verbose=False
        )

        self.is_trained = True
        print("Model training completed!")

    def get_user_recommendations(self, user_id, n_recommendations=10):
        if not self.is_trained:
            print("Model not trained yet!")
            return []

        user_id_map, _, item_id_map, _ = self.dataset.mapping()

        if user_id not in user_id_map:
            print(f"User {user_id} not found in dataset")
            return []

        user_x = user_id_map[user_id]
        n_items = len(item_id_map)
        item_ids = list(item_id_map.keys())

        scores = self.model.predict(
            user_ids=user_x,
            item_ids=list(range(n_items)),
            item_features=self.item_features_matrix
        )

        known_items = set(self.train_interactions.tocsr()[user_x].indices)

        recommendations = [
            (score, item_ids[internal_id])
            for internal_id, score in enumerate(scores)
            if internal_id not in known_items
        ]

        recommendations.sort(reverse=True)
        return self._format_recommendations(recommendations[:n_recommendations], 'prediction_score')

    def get_similar_books(self, book_id, n_recommendations=10):
        if not self.is_trained:
            print("Model not trained yet!")
            return []

        _, _, item_id_map, _ = self.dataset.mapping()

        if book_id not in item_id_map:
            print(f"Book ID {book_id} not found in dataset")
            return []

        item_x = item_id_map[book_id]
        item_embeddings = self.model.item_embeddings
        reverse_item_map = {v: k for k, v in item_id_map.items()}

        target_embedding = item_embeddings[item_x].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, item_embeddings)[0]

        similar_books = [
            (sim, reverse_item_map[internal_id])
            for internal_id, sim in enumerate(similarities)
            if internal_id != item_x and internal_id in reverse_item_map
        ]

        similar_books.sort(reverse=True)
        return self._format_recommendations(similar_books[:n_recommendations], 'similarity_score')

    def _format_recommendations(self, recommendations, score_type):
        result = []
        for score, book_id in recommendations:
            book_info = self.books_df[self.books_df['book_id'] == book_id]
            if not book_info.empty:
                result.append({
                    'book_id': book_id,
                    'title': book_info.iloc[0]['title'],
                    'author': book_info.iloc[0]['authors'],
                    'average_rating': book_info.iloc[0]['average_rating'],
                    score_type: score
                })
        return result

    def display_recommendations(self, recommendations, title, score_type):
        print("=" * 80)
        print(title)
        print("=" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['title']}")
            print(f"    Author: {rec['author']}")
            print(f"    Average Rating: {rec['average_rating']:.2f}")
            print(f"    {score_type.replace('_', ' ').title()}: {rec[score_type]:.4f}")
            print(f"    Book ID: {rec['book_id']}")
            print()

def main():
    try:
        print("Initializing recommendation system...")
        recommender = BookRecommendationSystem()
        
        print("Loading and processing data...")
        interactions, weights, train_weights = recommender.load_and_process_data()
        print(f"✓ Data loaded successfully!")
        print(f"  - Interactions shape: {interactions.shape}")
        print(f"  - Number of users: {len(recommender.processed_ratings['user_id'].unique())}")
        print(f"  - Number of books: {len(recommender.books_df)}")
        
        print("\nTraining model...")
        recommender.train_model(interactions, weights, train_weights)
        print("✓ Model training completed successfully!")
        
        print("\n" + "="*80)
        print("BOOK RECOMMENDATION SYSTEM - READY!")
        print("="*80)
        print("Instructions:")
        print("- Enter a user_id to get personalized recommendations")
        print("- Enter a book_id to get similar books")
        print("- Enter both to get both types of recommendations")
        print("- Type 'quit' to exit")
        print("="*80)

        while True:
            try:
                print("\nEnter your request:")
                user_input = input("User ID (or press Enter to skip): ").strip()
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break

                book_input = input("Book ID (or press Enter to skip): ").strip()
                if book_input.lower() == 'quit':
                    print("Goodbye!")
                    break

                user_id = int(user_input) if user_input.isdigit() else None
                book_id = int(book_input) if book_input.isdigit() else None

                if not user_id and not book_id:
                    print("⚠️  Please enter at least one ID (user_id or book_id)")
                    continue

                if user_id:
                    print(f"\n🔍 Generating recommendations for User ID: {user_id}")
                    user_recs = recommender.get_user_recommendations(user_id)
                    if user_recs:
                        recommender.display_recommendations(
                            user_recs,
                            f"PERSONALIZED RECOMMENDATIONS FOR USER {user_id}",
                            "prediction_score"
                        )
                    else:
                        print(f"❌ No recommendations found for User ID: {user_id}")

                if book_id:
                    target_book = recommender.books_df[recommender.books_df['book_id'] == book_id]
                    if not target_book.empty:
                        print(f"\n📖 Target Book: {target_book.iloc[0]['title']} by {target_book.iloc[0]['authors']}")
                    print(f"🔍 Finding books similar to Book ID: {book_id}")
                    similar_recs = recommender.get_similar_books(book_id)
                    if similar_recs:
                        recommender.display_recommendations(
                            similar_recs,
                            f"BOOKS SIMILAR TO BOOK ID {book_id}",
                            "similarity_score"
                        )
                    else:
                        print(f"❌ No similar books found for Book ID: {book_id}")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again...")

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please make sure books.csv and ratings.csv are in the correct location.")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install lightfm scikit-learn pandas numpy")
    except Exception as e:
        print(f"❌ Unexpected error during initialization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()