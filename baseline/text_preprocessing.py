import ast 
import pandas as pd


# def combine_text(row, columns):
#     """
#     Combines text from specified columns into a structured format.

#     Parameters:
#     - row (pd.Series): A single row of a DataFrame.
#     - columns (list): List of column names to include in the combined text.

#     Returns:
#     - str: A formatted string combining the selected column values.
#     Ex.: 
#     "Recipe name: (Chocolate Cake) | Minutes: (45) | Nutrition: (350 cal, 10g protein, 5g fat) | 
#     Steps: (Mix ingredients, Bake at 350F, Cool and serve) | Description: (A rich and moist 
#     chocolate cake) | Ingredients: (flour, sugar, cocoa, eggs, butter)"

#     Nutrition information :
#     (calories (#), 
#     total fat (PDV), 
#     sugar (PDV) , 
#     sodium (PDV) , 
#     protein (PDV) , 
#     saturated fat,
#     carbs,
#     """
#     combined_parts = []
#     nutrient_labels = ["calories", "total fat", "sugar", "sodium", "protein", "saturated fat", "carbs"]

#     for col in columns:
#         if col in row and pd.notnull(row[col]):
#             try:
#                 # Convert string representation of lists into actual lists
#                 value = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
#                 # If it's a list, join elements into a formatted string
#                 if isinstance(value, list):
#                     value = ", ".join(map(str, value))
#                 # Format text based on column name
#                 if col.lower() == "nutrition":
#                     formatted_text = f"{col.replace('_', ' ').capitalize()} {nutrient_labels}: ({value})"
#                 else:
#                     formatted_text = f"{col.replace('_', ' ').capitalize()}: ({value})"
#                 combined_parts.append(formatted_text)

#             except Exception:
#                 # Fallback as string
#                 combined_parts.append(f"{col.replace('_', ' ').capitalize()}: ({row[col]})")
#     return " | ".join(combined_parts)  # Join formatted parts with separators


def combine_text(row, columns):
    """
    Combines text from specified columns into a single structured string.
    
    Special handling for 'nutrition':
      - If the value is a scalar, simply convert it to a string.
      - If it is a list:
          * If it has exactly 7 numbers, format as:
             "Nutrition: ([a, b, c, d, e, f, g]) -> a calories, b total fat, c sugar, d sodium, e protein, f saturated fat, g carbs"
          * Otherwise, map over the available values using nutrient_labels (up to the number available).
    For all other columns, if the value is a string representation of a list,
    it is converted into an actual list and then joined.
    
    Returns:
      A formatted string.
    """
    combined_parts = []
    nutrient_labels = ["calories", "total fat", "sugar", "sodium", "protein", "saturated fat", "carbs"]
    
    for col in columns:
        if col in row and pd.notnull(row[col]):
            try:
                if col.lower() == "nutrition":
                    # For nutrition: if it's a string, try to parse it.
                    if isinstance(row[col], str):
                        value = ast.literal_eval(row[col])
                    else:
                        value = row[col]
                    # If value is numeric, output directly.
                    if isinstance(value, (int, float)):
                        formatted_text = f"{col.capitalize()}: ({value})"
                    elif isinstance(value, list):
                        n = min(len(value), len(nutrient_labels))
                        nutrient_list_str = ", ".join(map(str, value))
                        detailed = ", ".join([f"{value[i]} {nutrient_labels[i]}" for i in range(n)])
                        formatted_text = f"{col.capitalize()}: ([{nutrient_list_str}]) -> {detailed}"
                    else:
                        formatted_text = f"{col.capitalize()}: ({value})"
                else:
                    # For other columns, try to convert string representation of lists into actual lists.
                    if isinstance(row[col], str):
                        try:
                            value = ast.literal_eval(row[col])
                            if isinstance(value, list):
                                value = ", ".join(map(str, value))
                        except Exception:
                            value = row[col]
                    else:
                        value = row[col]
                    formatted_text = f"{col.replace('_', ' ').capitalize()}: ({value})"
            except Exception:
                formatted_text = f"{col.replace('_', ' ').capitalize()}: ({row[col]})"
            combined_parts.append(formatted_text)
    return " | ".join(combined_parts)



def get_combine_text(recipes_df, columns_to_combine):
    
    # Create a new column 'combined_text' with the select column
    recipes_df['combined_text'] = recipes_df.apply(lambda row: combine_text(row, columns_to_combine), axis=1)

    # Display a sample of the combined document.
    print("\nSample combined recipe text:")
    print(recipes_df[['id', 'combined_text']].head())
    return recipes_df