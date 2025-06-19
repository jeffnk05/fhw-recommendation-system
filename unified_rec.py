import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time

# Import your existing recommender classes/functions
# Assuming you've converted your scripts to importable modules
try:
    from lenskit_knn.knn_scorer import setup_recommender as setup_knn, get_recommendations_for_user as knn_recommend
    from lightfm_.lightfm_warp import BookRecommendationSystem
    from svd_surprise.surprise_engine import SVD, Dataset, Reader, train_test_split, recommend_for_user
except ImportError:
    st.error("Please make sure all recommender modules are properly installed and accessible.")

class UnifiedRecommendationSystem:
    def __init__(self):
        self.models = {}
        self.books_df = None
        self.ratings_df = None
        self.is_initialized = False
    
    @st.cache_data
    def load_data(_self):
        """Load and cache the data"""
        try:
            books_df = pd.read_csv('ALS/data/books.csv')
            ratings_df = pd.read_csv('ALS/data/ratings.csv')
            st.success("✅ Data loaded successfully")
            return books_df, ratings_df
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            return None, None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    
    def initialize_knn_model(self):
        """Initialize KNN model"""
        try:
            from lenskit.basic import topn
            from lenskit.knn import ItemKNNScorer
            from lenskit.batch import recommend
            from lenskit.data import from_interactions_df
            from lenskit.pipeline import topn_pipeline
            
            dataset = from_interactions_df(
                self.ratings_df, 
                user_col='user_id', 
                item_col='book_id', 
                rating_col='rating'
            )
            
            model_ii = ItemKNNScorer(
                max_nbrs=20,
                min_nbrs=3,
                min_sim=0.11
            )
            
            pipe_ii = topn_pipeline(model_ii)
            pipe_ii.train(dataset)
            
            self.models['knn'] = {'pipeline': pipe_ii}
            return True
        except Exception as e:
            st.error(f"Failed to initialize KNN model: {e}")
            return False
    
    def initialize_lightfm_model(self):
        """Initialize LightFM model"""
        try:
            from lightfm import LightFM
            from lightfm.data import Dataset
            from lightfm.cross_validation import random_train_test_split
            
            # Create simplified LightFM implementation
            dataset = Dataset()
            dataset.fit(
                users=self.ratings_df['user_id'].unique(),
                items=self.ratings_df['book_id'].unique()
            )
            
            # Create interactions matrix
            interactions, weights = dataset.build_interactions(
                [(row['user_id'], row['book_id'], row['rating']) 
                 for _, row in self.ratings_df.iterrows()]
            )
            
            train_interactions, _ = random_train_test_split(
                interactions, test_percentage=0.2, random_state=42
            )
            
            # Train model
            model = LightFM(loss='warp', random_state=42)
            model.fit(train_interactions, epochs=10)
            
            self.models['lightfm'] = {
                'model': model,
                'dataset': dataset,
                'interactions': train_interactions
            }
            return True
        except Exception as e:
            st.error(f"Failed to initialize LightFM model: {e}")
            return False
    
    def initialize_svd_model(self):
        """Initialize SVD model"""
        try:
            from surprise import Dataset, Reader, SVD
            from surprise.model_selection import train_test_split
            
            # Clean ratings data
            ratings_clean = self.ratings_df.copy()
            ratings_clean = ratings_clean[pd.to_numeric(ratings_clean['rating'], errors='coerce').notnull()]
            ratings_clean['rating'] = ratings_clean['rating'].astype(float)
            
            reader = Reader(rating_scale=(0, 5))
            data = Dataset.load_from_df(ratings_clean[['user_id', 'book_id', 'rating']], reader)
            trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
            
            algo = SVD(n_factors=20, random_state=42)
            algo.fit(trainset)
            
            self.models['svd'] = {'algo': algo, 'trainset': trainset}
            return True
        except Exception as e:
            st.error(f"Failed to initialize SVD model: {e}")
            return False
    
    @st.cache_resource
    def initialize_all_models(_self):
        """Initialize all models with caching"""
        success_count = 0
        
        with st.spinner("Initializing models..."):
            if _self.initialize_knn_model():
                st.success("✅ KNN model ready")
                success_count += 1
            
            if _self.initialize_lightfm_model():
                st.success("✅ LightFM model ready") 
                success_count += 1
            
            if _self.initialize_svd_model():
                st.success("✅ SVD model ready")
                success_count += 1
        
        return success_count > 0
    
    def get_knn_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get KNN recommendations"""
        if 'knn' not in self.models:
            return []
        
        try:
            from lenskit.batch import recommend
            
            pipe_ii = self.models['knn']['pipeline']
            recs = recommend(pipe_ii, [user_id], n_recommendations)
            
            results = []
            for user_key, item_list in recs:
                recommendations_df = item_list.to_df()
                recommendations_with_titles = recommendations_df.merge(
                    self.books_df[['book_id', 'title', 'authors']], 
                    left_on='item_id', 
                    right_on='book_id', 
                    how='left'
                )
                
                for _, row in recommendations_with_titles.head(n_recommendations).iterrows():
                    results.append({
                        'title': row.get('title', f"Book ID {row['item_id']}"),
                        'author': row.get('authors', 'Unknown'),
                        'score': row.get('score', 0.0),
                        'book_id': row.get('book_id', row['item_id'])
                    })
            
            return results
        except Exception as e:
            st.error(f"KNN recommendation error: {e}")
            return []
    
    def get_lightfm_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get LightFM recommendations"""
        if 'lightfm' not in self.models:
            return []
        
        try:
            model = self.models['lightfm']['model']
            dataset = self.models['lightfm']['dataset']
            
            user_id_map, _, item_id_map, _ = dataset.mapping()
            
            if user_id not in user_id_map:
                return []
            
            user_x = user_id_map[user_id]
            n_items = len(item_id_map)
            item_ids = list(item_id_map.keys())
            
            scores = model.predict(user_x, list(range(n_items)))
            
            # Get top recommendations
            top_items = np.argsort(scores)[::-1][:n_recommendations]
            
            results = []
            for internal_id in top_items:
                item_id = item_ids[internal_id]
                book_info = self.books_df[self.books_df['book_id'] == item_id]
                
                if not book_info.empty:
                    results.append({
                        'title': book_info.iloc[0]['title'],
                        'author': book_info.iloc[0].get('authors', 'Unknown'),
                        'score': float(scores[internal_id]),
                        'book_id': item_id
                    })
            
            return results
        except Exception as e:
            st.error(f"LightFM recommendation error: {e}")
            return []
    
    def get_svd_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get SVD recommendations"""
        if 'svd' not in self.models:
            return []
        
        try:
            algo = self.models['svd']['algo']
            trainset = self.models['svd']['trainset']
            
            # Get all items
            all_items = trainset._raw2inner_id_items.keys()
            
            # Get items user has already rated
            if user_id in trainset._raw2inner_id_users:
                rated_items = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
            else:
                rated_items = set()
            
            # Get candidates (unrated items)
            candidates = [iid for iid in all_items if iid not in rated_items]
            
            # Make predictions
            predictions = [algo.predict(user_id, iid) for iid in candidates[:1000]]  # Limit for performance
            top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations]
            
            results = []
            for pred in top_predictions:
                book_id = pred.iid
                book_info = self.books_df[self.books_df['book_id'].astype(str) == str(book_id)]
                
                if not book_info.empty:
                    title = book_info.iloc[0]['title']
                    author = book_info.iloc[0].get('authors', 'Unknown')
                else:
                    title = f"Book ID: {book_id}"
                    author = 'Unknown'
                
                results.append({
                    'title': title,
                    'author': author,
                    'score': pred.est,
                    'book_id': book_id
                })
            
            return results
        except Exception as e:
            st.error(f"SVD recommendation error: {e}")
            return []

def main():
    st.title("📚 Unified Book Recommendation System")
    st.markdown("---")
    
    # Initialize the system
    if 'recommender' not in st.session_state:
        st.session_state.recommender = UnifiedRecommendationSystem()
    
    recommender = st.session_state.recommender
    
    # Load data and initialize models
    if not recommender.is_initialized:
        books_df, ratings_df = recommender.load_data()
        if books_df is not None and ratings_df is not None:
            recommender.books_df = books_df
            recommender.ratings_df = ratings_df
            recommender.initialize_all_models()
            recommender.is_initialized = True
        else:
            st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("🎛️ Configuration")
    
    # Check which models are available
    available_models = []
    if 'knn' in recommender.models:
        available_models.append("KNN (Item-based)")
    if 'lightfm' in recommender.models:
        available_models.append("LightFM (WARP)")
    if 'svd' in recommender.models:
        available_models.append("SVD (Matrix Factorization)")
    
    if not available_models:
        st.error("No models are available. Please check your installation.")
        st.stop()
    
    selected_model = st.sidebar.selectbox(
        "Select Recommendation Model",
        available_models,
        help="Choose which algorithm to use for recommendations"
    )
    
    # Parameters
    user_id = st.sidebar.text_input(
        "User ID",
        value="234",
        help="Enter the user ID for recommendations"
    )
    
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=10,
        help="How many book recommendations to show"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 Recommendations using {selected_model}")
        
        if st.button("🔍 Get Recommendations", type="primary"):
            if not user_id:
                st.warning("Please enter a user ID")
                return
            
            with st.spinner(f"Generating recommendations using {selected_model}..."):
                # Get recommendations based on selected model
                if "KNN" in selected_model:
                    try:
                        user_id_int = int(user_id)
                        recommendations = recommender.get_knn_recommendations(user_id_int, n_recommendations)
                    except ValueError:
                        st.error("User ID must be a number for KNN model")
                        return
                elif "LightFM" in selected_model:
                    try:
                        user_id_int = int(user_id)
                        recommendations = recommender.get_lightfm_recommendations(user_id_int, n_recommendations)
                    except ValueError:
                        st.error("User ID must be a number for LightFM model")
                        return
                elif "SVD" in selected_model:
                    recommendations = recommender.get_svd_recommendations(user_id, n_recommendations)
                else:
                    recommendations = []
                
                # Display recommendations
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations for User {user_id}")
                    
                    # Create a nice display for recommendations
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"#{i} - {rec['title']}", expanded=i <= 3):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.write(f"**Author:** {rec['author']}")
                                st.write(f"**Book ID:** {rec['book_id']}")
                            
                            with col_b:
                                st.metric("Score", f"{rec['score']:.3f}")
                else:
                    st.warning(f"No recommendations found for User {user_id}")
    
    with col2:
        st.header("ℹ️ Model Information")
        
        # Display model info
        if "KNN" in selected_model:
            st.info("""
            **KNN (Item-based Collaborative Filtering)**
            - Uses item-item similarities
            - Good for finding similar books
            - Works well with sparse data
            """)
        elif "LightFM" in selected_model:
            st.info("""
            **LightFM with WARP Loss**
            - Hybrid approach (collaborative + content)
            - Uses matrix factorization
            - Good for implicit feedback
            """)
        elif "SVD" in selected_model:
            st.info("""
            **SVD (Singular Value Decomposition)**
            - Matrix factorization approach
            - Finds latent factors
            - Handles rating prediction well
            """)
        
        # Show model status
        st.subheader("📈 Model Status")
        for model_name in ['knn', 'lightfm', 'svd']:
            if model_name in recommender.models:
                st.success(f"✅ {model_name.upper()} - Ready")
            else:
                st.error(f"❌ {model_name.upper()} - Not available")
        
        # Show data info
        if recommender.books_df is not None:
            st.subheader("📚 Data Info")
            st.write(f"Books: {len(recommender.books_df):,}")
            st.write(f"Ratings: {len(recommender.ratings_df):,}")
            st.write(f"Users: {recommender.ratings_df['user_id'].nunique():,}")

if __name__ == "__main__":
    main()