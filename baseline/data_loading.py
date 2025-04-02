
import pandas as pd
import ast
from pathlib import Path

def load_data_dfs(RAW_recepies_path, RAW_interactions_path):

    recipes_df = pd.read_csv(RAW_recepies_path)
    interactions_df = pd.read_csv(RAW_interactions_path)
    print(f"Recipes Data: {recipes_df.shape[0]} rows.")
    print(f"Interactions Data: {interactions_df.shape[0]} rows.")

    # Drop rows with any null values
    recipes_df.dropna(inplace=True)
    interactions_df.dropna(inplace=True)
    # Reset index after dropping rows
    recipes_df.reset_index(drop=True, inplace=True)
    interactions_df.reset_index(drop=True, inplace=True)

    print(f"Recipes Data: {recipes_df.shape[0]} rows after cleaning.")
    print(f"Interactions Data: {interactions_df.shape[0]} rows after cleaning.")
    return recipes_df, interactions_df


def load_users_token_df(path: Path) -> pd.DataFrame:
    """
    Loads the PP_users.csv file and returns a DataFrame with columns:
      - 'u': user ID
      - 'items': string representation of recipe IDs
      - 'ratings': string representation of ratings
    """
    users_df = pd.read_csv(path)
    print("PP_users.csv columns:", users_df.columns.tolist())
    print("PP_users.csv shape:", users_df.shape)

    # Convert the string representations into actual lists.
    users_df['items'] = users_df['items'].apply(ast.literal_eval)
    users_df['ratings'] = users_df['ratings'].apply(ast.literal_eval)
    return users_df


def load_recipes_token_df(path: Path) -> pd.DataFrame:
    """
    Loads the PP_recipes.csv file and returns a DataFrame with columns:
      - 'id': recipe ID
      - 'name_tokens': string representation of tokens
      - 'ingredient_tokens': string representation of ingredient tokens
      - 'steps_tokens': string representation of steps
    """
    recipes_df = pd.read_csv(path)
    print("PP_recipes.csv columns:", recipes_df.columns.tolist())
    print("PP_recipes.csv shape:", recipes_df.shape)
    return recipes_df