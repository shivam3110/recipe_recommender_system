import pandas as pd
from scipy.sparse import csr_matrix

def compute_bayesian_average(interactions_df, recipes_df):
    """
    Computes Bayesian average ratings for recipes based on user interactions.

    Parameters:
    - interactions_df (pd.DataFrame): DataFrame containing user interactions with 'recipe_id' and 'rating' columns.
    - recipes_df (pd.DataFrame): DataFrame containing recipes with an 'id' column.

    Returns:
    - pd.DataFrame: Updated recipes_df with a new column 'bayesian_avg'.
    """
    print(interactions_df.columns.tolist())
    #########################################
    #  Compute Bayesian Average Ratings for Recipes
    #########################################
    # Aggregate ratings per recipe
    agg = interactions_df.groupby('recipe_id').agg(
        v=('rating', 'count'),  # Number of ratings
        R=('rating', 'mean')    # Mean rating
    ).reset_index()

    # Compute global average rating C
    C = (agg['v'] * agg['R']).sum() / agg['v'].sum()
    # Choose smoothing parameter m (median number of ratings)
    m = agg['v'].median()
    # Compute Bayesian average rating
    agg['bayesian_avg'] = (agg['v'] / (agg['v'] + m)) * agg['R'] + (m / (agg['v'] + m)) * C

    # Merge Bayesian ratings into recipes_df
    recipes_df = recipes_df.merge(agg[['recipe_id', 'bayesian_avg']], left_on='id', right_on='recipe_id', how='left')
    # Fill missing Bayesian averages with global average rating C
    recipes_df['bayesian_avg'] = recipes_df['bayesian_avg'].fillna(C)

    print("\nSample recipes with Bayesian average rating:")
    print(recipes_df[['id', 'bayesian_avg']].head())
    return recipes_df



def build_interactions(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the 'items' and 'ratings' columns from users_df into
    a row per (user_id, recipe_id, rating).
    Returns an interactions DataFrame with columns ['user_id', 'recipe_id', 'rating'].
    """
    interaction_list = []
    for _, row in users_df.iterrows():
        user_id = row['u']
        for recipe_id, rating in zip(row['items'], row['ratings']):
            interaction_list.append({'user_id': user_id, 'recipe_id': recipe_id, 'rating': rating})
    interactions_df = pd.DataFrame(interaction_list)
    print("\nSample interactions:")
    print(interactions_df.head())
    print("Total interactions:", interactions_df.shape[0])
    return interactions_df


def build_user_item_matrix(interactions_df: pd.DataFrame):
    """
    Creates a sparse CSR user-item matrix from the interactions DataFrame.
    Returns (user_item_sparse, n_users, n_recipes).
    """
    n_users = interactions_df['user_id'].max() + 1
    n_recipes = interactions_df['recipe_id'].max() + 1
    print(f"\nNumber of users: {n_users}, Number of recipes: {n_recipes}")

    row_indices = interactions_df['user_id'].values
    col_indices = interactions_df['recipe_id'].values
    ratings = interactions_df['rating'].values

    user_item_sparse = csr_matrix((ratings, (row_indices, col_indices)), shape=(n_users, n_recipes))
    print("User-Item Sparse Matrix shape:", user_item_sparse.shape)
    return user_item_sparse, n_users, n_recipes
