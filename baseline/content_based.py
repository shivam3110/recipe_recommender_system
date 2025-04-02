# baseline/content_based.py
import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def combine_recipe_text(row):
    """
    Combine name_tokens, ingredient_tokens, steps_tokens into one text doc.
    """
    name_text = " ".join(str(t) for t in row['name_tokens'])
    ingredient_text = " ".join(" ".join(str(tok) for tok in sub) for sub in row['ingredient_tokens'])
    calorie_text = " ".join(str( row['calorie_level']))
    return f"Recipe Name: {name_text} | Ingridents: {ingredient_text} | Calorie: {calorie_text}"


def build_recipe_embeddings(recipes_df: pd.DataFrame, 
                            model_name='all-MiniLM-L6-v2'):
    """
    Creates a column 'combined_text' in recipes_df, then uses the specified SentenceTransformer
    to compute embeddings for each recipe. Returns a dict {recipe_id: embedding}.
    """
    # Combine text fields
    recipes_df['combined_text'] = recipes_df.apply(combine_recipe_text, axis=1)

    # Initialize model
    embedding_model = SentenceTransformer(model_name)
    recipe_ids = recipes_df['id'].tolist()
    recipe_texts = recipes_df['combined_text'].tolist()

    # Compute embeddings
    embeddings = embedding_model.encode(recipe_texts, batch_size=64, 
                                        show_progress_bar=True, 
                                        convert_to_tensor=True)

    # Build dict
    recipe_embeddings = {rid: emb for rid, emb in zip(recipe_ids, embeddings)}
    return recipe_embeddings


def build_user_profiles_content(interactions_df: pd.DataFrame, 
                                recipe_embeddings: dict, 
                                rating_threshold=4):
    """
    For each user, average the embeddings of recipes they rated >= threshold.
    Returns a dict {user_id: embedding}.
    """
    user_profiles = {}
    unique_users = interactions_df['user_id'].unique()

    for user in tqdm(unique_users, desc="Building user profiles (content-based)"):
        user_data = interactions_df[interactions_df['user_id'] == user]
        liked_recipes = user_data[user_data['rating'] >= rating_threshold]['recipe_id'].tolist()

        embeddings_list = [recipe_embeddings[rid] for rid in liked_recipes if rid in recipe_embeddings]
        if embeddings_list:
            stacked = torch.stack(embeddings_list)
            user_profiles[user] = stacked.mean(dim=0)
    return user_profiles


def recommend_content_based(user_id: int, 
                            user_profiles: dict, 
                            recipe_embeddings: dict, 
                            interactions_df: pd.DataFrame, 
                            top_k=10):
    """
    Recommends recipes for a user by computing cosine similarity between user profile and each recipe embedding.
    Excludes recipes the user has already rated.
    """
    if user_id not in user_profiles:
        return []

    profile = user_profiles[user_id]
    scores = {}
    already_rated = set(interactions_df[interactions_df['user_id'] == user_id]['recipe_id'])
    for rid, emb in recipe_embeddings.items():
        if rid in already_rated:
            continue
        sim = F.cosine_similarity(profile, emb, dim=0)
        scores[rid] = sim.item()

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return recommended
