"""
python -m baseline.run_token_recommender
"""

# baseline/main.py
import pandas as pd
from pathlib import Path

from baseline.config import PP_users_path, PP_recipes_path, RAW_recepies_path
from baseline.data_loading import load_users_token_df, load_recipes_token_df
from baseline.utils import build_interactions, build_user_item_matrix
from baseline.recommendation_engine import recommend_collaborative, recommend_hybrid
from baseline.content_based import (
    build_recipe_embeddings,
    build_user_profiles_content,
    recommend_content_based
)


def map_recommendations_to_details(recommendations, recipes_df, detail_columns=None):
    """
    Given a list of (recipe_id, score) tuples and a DataFrame of recipes,
    returns a sorted DataFrame containing the specified columns plus 'score'.

    Parameters:
      recommendations: List[Tuple[int, float]]
        A list of (recipe_id, score) tuples, e.g. [(135961, 0.4032), (134610, 0.3209), ...]
      recipes_df: pd.DataFrame
        The DataFrame loaded from RAW_recipes.csv (must contain 'id' column).
      detail_columns: List[str]
        Which columns from recipes_df to include (besides 'id' and 'score').
        For example: ['name', 'minutes', 'submitted', 'nutrition', 'n_steps', 'steps', 'n_ingredients'].

    Returns:
      pd.DataFrame:
        A DataFrame with columns [id, score, ... detail_columns ...], sorted by 'score' descending.
    """
    # Default columns if none provided.
    if detail_columns is None:
        detail_columns = ['name', 'minutes', 'submitted', 'nutrition', 'n_steps', 'steps', 'n_ingredients']

    # Convert the list of tuples into a small DataFrame.
    rec_df = pd.DataFrame(recommendations, columns=['id', 'score'])
    # Merge with recipes_df on 'id' to retrieve details.
    merged_df = rec_df.merge(recipes_df, how='left', left_on='id', right_on='id')
    # Reorder columns: first 'id', 'score', then the detail columns (if they exist in merged_df).
    final_cols = ['id', 'score'] + [col for col in detail_columns if col in merged_df.columns]
    merged_df = merged_df[final_cols]
    # Sort by 'score' in descending order.
    merged_df = merged_df.sort_values(by='score', ascending=False)
    return merged_df

def process_result(columns_list, recommendations, raw_recipe_df):
    recommended_details = map_recommendations_to_details(recommendations,
                                                        raw_recipe_df,
                                                        detail_columns=columns_list
                                                        )
    print(recommended_details)
    return 


def main():
    # 1. Load Data
    users_df = load_users_token_df(PP_users_path)
    recipes_df = load_recipes_token_df(PP_recipes_path)

    raw_recipe_df = pd.read_csv(RAW_recepies_path)
    columns_list = ['name','minutes','submitted','nutrition','n_steps','steps','n_ingredients']

    # 2. Build Interactions
    interactions_df = build_interactions(users_df)
    user_item_sparse, n_users, n_recipes = build_user_item_matrix(interactions_df)

    # 3. Collaborative Recs
    sample_user = 0
    print("\n--- Collaborative Filtering Recommendations for User", sample_user, "---")
    collab_recs = recommend_collaborative(sample_user, 
                                          user_item_sparse, 
                                          interactions_df, 
                                          n_recipes, 
                                          top_k=10)
    print(collab_recs)
    process_result(columns_list, collab_recs, raw_recipe_df)


    # 4. Content-Based Recs
    #    Build embeddings and user profiles
    recipe_embeddings = build_recipe_embeddings(recipes_df, model_name='all-MiniLM-L6-v2')
    user_profiles = build_user_profiles_content(interactions_df, recipe_embeddings, rating_threshold=4)
    print("\n--- Content-Based Recommendations for User", sample_user, "---")
    content_recs = recommend_content_based(sample_user, 
                                           user_profiles, 
                                           recipe_embeddings, 
                                           interactions_df, 
                                           top_k=10)
    print(content_recs)
    process_result(columns_list, content_recs, raw_recipe_df)

    # 5. Hybrid
    print("\n--- Hybrid Recommendations for User", sample_user, "---")
    hybrid_results = recommend_hybrid(
        user_id=sample_user,
        recommend_collaborative_fn=lambda uid, top_k: recommend_collaborative(uid, user_item_sparse, interactions_df, n_recipes, top_k),
        recommend_content_fn=lambda uid, top_k: recommend_content_based(uid, user_profiles, recipe_embeddings, interactions_df, top_k),
        top_k=10, alpha=0.5
    )
    print(hybrid_results)
    process_result(columns_list, hybrid_results, raw_recipe_df)
    #TODO:EValuation metric: BPE PPL BLEU-1 BLEU-4 ROUGE-L D-1 (%) D-2 (%) UMA MRR PP (%)
    # Use token test_set to evaluate the recommender system
    # Compare metrics with benchmark (pubication) 


if __name__ == "__main__":
    main()
