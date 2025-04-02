"""

python -m baseline.recommend_recipe 

"""

import ast
import pandas as pd

from baseline.config import DATA_DIR, EMBDDINGS_DIR, RECIPE_EMBD_DIR, USER_PROFILE_EMBD_DIR
from baseline.config import RAW_recepies_path, RAW_interactions_path
from baseline.text_preprocessing import get_combine_text
from baseline.utils import compute_bayesian_average
from baseline.recommendation_engine import recommend_recipes
from baseline.data_loading import load_data_dfs


def main():

    #########################################
    # 1. Load the Data
    #########################################
    # Adjust file paths in baseline/config.py.
    recipes_df, interactions_df = load_data_dfs(RAW_recepies_path, RAW_interactions_path)

    #########################################
    # 2. Compute Bayesian Average Ratings for Recipes
    #########################################   
    recipes_df = compute_bayesian_average( interactions_df, recipes_df)

    #########################################
    # 3. Process Recipe text information for content
    #########################################
    # List of columns the user wants to include in combined_text
    # columns_to_combine = ['name', 'minutes', 'nutrition', 'steps', 'description', 'ingredients']
    columns_to_combine = ['name', 'ingredients', 'nutrition']
    recipes_df = get_combine_text(recipes_df, columns_to_combine)


    #########################################
    # 4. Select Recommender function
    #########################################
    recommendation_types = ['user_based', 'content_based', 'hybrid']
    recommendation_type = 'user_based'
    try:
        if recommendation_type in recommendation_types:
            print("User selected Recommendation type: ", recommendation_type)
        else:
            raise ValueError(f"Invalid recommendation type: {recommendation_type}.")
    except Exception as e:
        print("Error:", e)


    #########################################
    # 5. Generate Recommendations
    #########################################
    rating_threshold = 4
    top_k=10
    user_id = interactions_df['user_id'].iloc[0]

    # print(recipes_df)
    # print(recipes_df['combined_text'][0])
    recommendations = recommend_recipes(recommendation_type, 
                                            recipes_df,
                                            interactions_df,
                                            rating_threshold,
                                            user_id,                                            
                                            top_k,                                           
                                            alpha=0.5,
                                            )
    print("\n--- Recommended Recipes for User", user_id, "---")
    print(recommendations)
    # recommendations

    return recommendations


if __name__ == '__main__':
    main()