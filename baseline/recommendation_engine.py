import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from baseline.user_profile_recommender import get_user_based_recommendation_tfidf_baysian_avg


def recommend_collaborative(user_id: int, 
                            user_item_sparse, 
                            interactions_df,
                            n_recipes: int, 
                            top_k=10):
    """
    Recommend recipes for a given user using collaborative filtering.
    Computes cosine similarity on the fly between the target user's vector and all user vectors.
    Then, aggregates ratings from similar users (weighted by similarity) to score recipes.
    Excludes recipes already rated by the user.
    """
    # Get target user's vector (1 x n_recipes)
    user_vec = user_item_sparse.getrow(user_id)
    # Compute cosine similarity between this user and all users.
    sim_scores = cosine_similarity(user_vec, user_item_sparse).flatten()

    # Compute a weighted score for each recipe.
    weighted_scores = user_item_sparse.T.dot(sim_scores)
    # Normalize by the sum of similarity scores.
    denom = sim_scores.sum() + 1e-8
    weighted_scores = weighted_scores / denom

    # Exclude recipes already rated by the user.
    user_rated = set(interactions_df[interactions_df['user_id'] == user_id]['recipe_id'])
    candidate_indices = [i for i in range(n_recipes) if i not in user_rated]

    candidate_scores = {recipe: weighted_scores[recipe] for recipe in candidate_indices}
    recommended = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return recommended


# baseline/hybrid.py
def recommend_hybrid(user_id: int,
                     recommend_collaborative_fn,
                     recommend_content_fn,
                     top_k=10,
                     alpha=0.5):
    """
    Hybrid recommender that combines scores from collaborative and content-based approaches.
    alpha: weight for collaborative filtering
    (1 - alpha) for content-based filtering.
    """
    collab_recs = recommend_collaborative_fn(user_id, top_k=top_k*2)
    content_recs = recommend_content_fn(user_id, top_k=top_k*2)

    # Convert to dict
    collab_dict = dict(collab_recs)
    content_dict = dict(content_recs)

    combined_scores = {}
    candidate_recipes = set(collab_dict.keys()).union(set(content_dict.keys()))
    for rid in candidate_recipes:
        c_score = collab_dict.get(rid, 0)
        ct_score = content_dict.get(rid, 0)
        combined_scores[rid] = alpha * c_score + (1 - alpha) * ct_score

    hybrid_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return hybrid_recs


def recommend_recipes(recommendation_type,                        
                        recipes_df,
                        interactions_df,
                        rating_threshold,
                        user_id = None,
                        top_k=10, 
                        alpha=0.5,
                        ):

    if recommendation_type == "user_based" and user_id:
        print('Functions user_based: user_based_recommendations_tfidf_baysian_avg')
        rec_df = get_user_based_recommendation_tfidf_baysian_avg(user_id,
                                  recipes_df,
                                  interactions_df,
                                  rating_threshold,
                                  top_k, 
                                  alpha,
                                  )
    elif recommendation_type == "content_based":
        print("Content-based filtering is not yet implemented.")
    elif recommendation_type == "hybrid":
        print("Hybrid recommendation is not yet implemented")
    else:
        print(f"Invalid recommendation type: {recommendation_type}")
    return rec_df