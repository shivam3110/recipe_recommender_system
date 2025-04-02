from pathlib import Path

DATA_DIR = Path(r"D:\UKW_work\code\recipe_recommender_system\data\food_com_GeniusKitchen")
EMBDDINGS_DIR = Path(r'D:\UKW_work\code\recipe_recommender_system\embeddings')
RECIPE_EMBD_DIR = EMBDDINGS_DIR / 'recipes_embeddings'
USER_PROFILE_EMBD_DIR = EMBDDINGS_DIR / 'user_profile_embeddings'

RAW_recepies_path = DATA_DIR / 'RAW_recipes.csv'
RAW_interactions_path =  DATA_DIR / 'RAW_interactions.csv'

PP_users_path = DATA_DIR / "PP_users.csv"
PP_recipes_path = DATA_DIR / "PP_recipes.csv"

token_interactions_train_path = DATA_DIR / 'interactions_train.csv'
token_interactions_val_path = DATA_DIR / 'interactions_validation.csv'
token_interactions_test_path = DATA_DIR / 'interactions_test.csv'