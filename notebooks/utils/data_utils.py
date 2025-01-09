import pandas as pd
import joblib
import re

from sklearn.preprocessing import StandardScaler
from langdetect import detect

from utils.general_utils import load_categories_from_file


def load_data(split, data_type):
    data = pd.read_csv(f'../data/raw_data/{split}_{data_type}.csv')
    return data


def fill_na(df, column, value):
    df[column] = df[column].fillna(value)


def to_lower(df, column):
    df[column] = df[column].str.lower()
    return df


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


def clean_text(text):
    if not text or text is None:
        return ''
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text


def replace_multiple_spaces(df, column):
    df[column] = df[column].str.strip().str.replace(r'\s+', ' ', regex=True)
    return df


def encode_categorical_to_one_hot(df, column, drop_original=True):
    categories = load_categories_from_file(f'../data/processed_data/{column}_categories.json')

    # Filter the column to include only categories present in the known categories
    df[column] = df[column].apply(lambda x: x if x in categories else None)

    # Create one-hot encoding for the input dataset (validation/test)
    df_one_hot = pd.get_dummies(df[column], prefix=column, dtype=float)

    # Ensure all train categories exist in the encoded test/validation set
    for category in categories:
        col_name = f"{column}_{category}"
        if col_name not in df_one_hot.columns:
            df_one_hot[col_name] = 0.0

    # Align columns to match train categories order
    df_one_hot = df_one_hot[[f"{column}_{cat}" for cat in categories]]

    # Concatenate the one-hot encoded columns with the original dataframe
    df = pd.concat([df.reset_index(drop=True), df_one_hot.reset_index(drop=True)], axis=1)

    # Drop the original column if specified
    if drop_original:
        df = df.drop(columns=[column])

    return df


def scaler_fit_and_save(df, column):
    scaler = StandardScaler()
    scaler.fit(df[column].values.reshape(-1, 1))
    joblib.dump(scaler, f"./assets/scaler_{column}.pkl")


def scale_range(df, column):
    scaler = joblib.load(f"./assets/scaler_{column}.pkl")
    df[column] = scaler.transform(df[column].values.reshape(-1, 1))
    return df


def merge_users_reviews_by_matches(df_users, df_reviews, split):
    df_matches = load_data(split, 'matches')

    df_merged = pd.merge(df_matches, df_reviews, on='review_id', how='inner')
    df_merged = df_merged.drop(columns=['accommodation_id_y'], axis=1)
    df_merged = df_merged.rename(columns={'accommodation_id_x': 'accommodation_id'})

    df_merged = pd.merge(df_merged, df_users, on='user_id', how='inner')
    df_merged = df_merged.drop(columns=['accommodation_id_y'], axis=1)
    df_merged = df_merged.rename(columns={'accommodation_id_x': 'accommodation_id'})

    df_merged.to_csv(f'../data/processed_data/processed_{split}.csv', index=False)
