import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_user_profiles( tfidf_matrix, recipes_df, interactions_df, rating_threshold=4):
    """
    Computes user profiles using the TF-IDF matrix of recipe text.
    
    Parameters:
    - interactions_df (pd.DataFrame): User-recipe interaction dataframe.
    - tfidf_matrix (scipy.sparse matrix): TF-IDF sparse matrix of combined recipe text.
    - recipe_id_to_index (dict): Mapping of recipe_id to row index in the TF-IDF matrix.
    - rating_threshold (int): Minimum rating to consider a recipe "liked" (default: 4).
    
    Returns:
    - dict: A dictionary mapping user_id to a PyTorch tensor representing their profile.
    """
    user_profiles = {}
    # Get unique users
    unique_users = interactions_df['user_id'].unique()

    # Create a mapping from recipe id to its TF-IDF vector (as a row in tfidf_matrix).
    # Note: recipes_df index corresponds to the row in tfidf_matrix.
    recipe_id_to_index = {rid: idx for idx, rid in enumerate(recipes_df['id'])}

    # Loop over each user and compute profile
    for user, group in tqdm(interactions_df.groupby('user_id'), total=len(unique_users), desc="Building user profiles"):
        # Get the recipe IDs rated highly by the user
        liked_recipe_ids = group[group['rating'] >= rating_threshold]['recipe_id'].tolist()        
        # Map recipe IDs to TF-IDF matrix indices
        indices = [recipe_id_to_index[rid] for rid in liked_recipe_ids if rid in recipe_id_to_index]
        
        if indices:
            # Extract the liked recipe rows from the TF-IDF matrix
            liked_sparse = tfidf_matrix[indices, :]            
            # Compute mean vector for the user
            user_profile_sparse = liked_sparse.sum(axis=0) / len(indices)            
            # Convert to a dense array
            user_profile_dense = np.asarray(user_profile_sparse).flatten()

            # # Convert to a PyTorch tensor
            user_profiles[user] = torch.tensor(user_profile_dense, dtype=torch.float32).reshape(1, -1)
            # Convert to a PyTorch tensor using from_numpy.
            # user_profiles[user] = torch.from_numpy(user_profile_dense).float().reshape(1, -1)

    print("\nComputed user profiles for", len(user_profiles), "users.")    
    return user_profiles



##################################################################################
#  Recommendation Function: Content-Based(TF-IDF) + Bayesian Rating
##################################################################################
def generate_user_based_recommendations_tfidf_baysian_avg(user_id, 
                                    user_profiles,
                                    recipes_df, 
                                    interactions_df, 
                                    tfidf_matrix,
                                    top_k=10, alpha=0.5):
    """
    For a given user, recommend recipes by combining:
      - Content similarity: cosine similarity between the user's profile (TF-IDF based) and each recipe.
      - Bayesian average: the adjusted quality score.
    
    We compute:
       final_score = alpha * (normalized similarity) + (1 - alpha) * (normalized bayesian rating)
    
    Normalization is done over all recipes. Recipes the user has already interacted with are excluded.
    
    Parameters:
      user_id (int): The ID of the user.
      top_k (int): Number of recipes to return.
      alpha (float): Weight for the content similarity score.
    
    Returns:
      DataFrame: Recommended recipes with columns: id, name, ingredients, bayesian_avg, final_score.
    """
    # If the user doesn't have a computed profile, return top recipes by Bayesian average.
    if user_id not in user_profiles:
        print("User has no high-rated recipes; returning top recipes by Bayesian average.")
        top = recipes_df.sort_values(by='bayesian_avg', ascending=False).head(top_k)
        return top[['id', 'bayesian_avg']]
    
    # Get user profile (stored as a PyTorch tensor) and convert it to NumPy.
    user_profile = user_profiles[user_id]  # shape: (1, feature_dim)
    user_profile_np = user_profile.cpu().numpy()
    
    # Compute cosine similarity between the user profile and all recipes in the TF-IDF matrix.
    # Note: tfidf_matrix_combined_text remains in sparse format.
    similarities = cosine_similarity(user_profile_np, tfidf_matrix).flatten()  # shape: (n_recipes,)
    
    # Normalize similarity scores between 0 and 1.
    if similarities.max() > similarities.min():
        norm_sim = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    else:
        norm_sim = similarities
    
    # Extract Bayesian average scores from recipes_df.
    bayes_scores = recipes_df['bayesian_avg'].values.astype(np.float32)
    if bayes_scores.max() > bayes_scores.min():
        norm_bayes = (bayes_scores - bayes_scores.min()) / (bayes_scores.max() - bayes_scores.min())
    else:
        norm_bayes = bayes_scores
    
    # Combine the two signals via weighted sum.
    final_scores = alpha * norm_sim + (1 - alpha) * norm_bayes
    
    # Build a DataFrame with recipe ids and the computed final score.
    rec_df = pd.DataFrame({
        'id': recipes_df['id'],
        'final_score': final_scores
    })    
    # Exclude recipes already seen by the user.
    already_seen = set(interactions_df[interactions_df['user_id'] == user_id]['recipe_id'])
    rec_df = rec_df[~rec_df['id'].isin(already_seen)]
    
    # Sort the recipes by final_score (highest first) and select the top_k.
    rec_df = rec_df.sort_values(by='final_score', ascending=False).head(top_k)
    
    # Merge additional recipe details for display.
    rec_df = rec_df.merge(recipes_df[['id', 'name', 'ingredients', 'bayesian_avg']], on='id', how='left')
    return rec_df


def get_user_based_recommendation_tfidf_baysian_avg(user_id,
                                  recipes_df,
                                  interactions_df,
                                  rating_threshold,
                                  top_k=10, 
                                  alpha=0.5,
                                  ):
    
    vectorizer_combined_text = TfidfVectorizer(stop_words='english', max_features=10000)
    # Fit and transform the ingredients documents.
    tfidf_matrix = vectorizer_combined_text.fit_transform(recipes_df['combined_text'])
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

    user_profiles = compute_user_profiles( tfidf_matrix, 
                                          recipes_df, 
                                          interactions_df, 
                                          rating_threshold)
    
    rec_df = generate_user_based_recommendations_tfidf_baysian_avg(user_id, 
                                        user_profiles,
                                        recipes_df, 
                                        interactions_df, 
                                        tfidf_matrix,
                                        top_k, alpha)
    return rec_df