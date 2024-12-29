import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from tqdm import tqdm
import time


def load_and_preprocess_data(reviews_path, users_path, matches_path):
    """
    Load and preprocess review, user, and match data.
    """
    start_time = time.time()
    print("Loading data...")
    reviews = pd.read_csv(reviews_path)
    users = pd.read_csv(users_path)
    matches = pd.read_csv(matches_path)

    print("Preprocessing data...")
    reviews['review_title'] = reviews['review_title'].fillna("No Title")
    reviews['review_positive'] = reviews['review_positive'].fillna("No Positive Review")
    reviews['review_negative'] = reviews['review_negative'].fillna("No Negative Review")

    users['guest_country'] = users['guest_country'].fillna("Unknown")
    users['room_nights'] = users['room_nights'].apply(lambda x: min(x, 30))

    data = pd.merge(matches, reviews, on='review_id', how='inner')
    data = pd.merge(data, users, on='user_id', how='inner')
    data['text'] = data['review_title'] + " " + data['review_positive'] + " " + data['review_negative']

    elapsed_time = time.time() - start_time
    print(f"Data loading and preprocessing completed in {elapsed_time:.2f} seconds.")
    return data


def split_data_by_users(data, test_size=0.2, random_state=42):
    """
    Split the data such that users in the test set are not in the training set.
    """
    start_time = time.time()
    print("Splitting data by users...")
    unique_users = data['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)

    train_data = data[data['user_id'].isin(train_users)]
    test_data = data[data['user_id'].isin(test_users)]

    elapsed_time = time.time() - start_time
    print(f"Data split completed in {elapsed_time:.2f} seconds.")
    return train_data, test_data


def prepare_features(data, vectorizer=None):
    """
    Prepare text, numeric, and categorical features from the dataset.
    """
    start_time = time.time()
    print("Preparing features...")

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)

    # Text features
    X_text = vectorizer.fit_transform(tqdm(data['text'], desc="Processing text features"))

    # Numeric features
    numeric_features = ['review_score', 'room_nights', 'accommodation_score', 'accommodation_star_rating',
                        'location_is_ski', 'location_is_beach', 'location_is_city_center']
    for col in numeric_features:
        if col not in data.columns:
            data[col] = 0
    X_numeric = data[numeric_features].fillna(0).values

    # Categorical features
    categorical_features = pd.get_dummies(data[['guest_type', 'guest_country', 'accommodation_type']], drop_first=True)

    elapsed_time = time.time() - start_time
    print(f"Feature preparation completed in {elapsed_time:.2f} seconds.")
    return X_text, X_numeric, categorical_features, vectorizer


def combine_features(X_text, X_numeric, X_categorical):
    """
    Combine text, numeric, and categorical features into a single matrix.
    """
    return hstack([X_text, X_numeric, X_categorical])


def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    """
    print("Training the model...")
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    with tqdm(total=len(y_train), desc="Training progress") as pbar:
        for _ in range(1):  # Simulating iteration-based progress (for a single fit operation, you can replace this)
            model.fit(X_train, y_train)
            pbar.update(len(y_train))

    elapsed_time = time.time() - start_time
    print(f"Model training completed in {elapsed_time:.2f} seconds.")
    return model


def calculate_mrr_at_k(y_true, y_pred, k=10):
    """
    Calculate Mean Reciprocal Rank at K (MRR@K).
    """
    print("Calculating MRR@10...")
    reciprocal_ranks = []

    for true_review, pred_reviews in tqdm(zip(y_true, y_pred), total=len(y_true), desc="Evaluating MRR@10"):
        try:
            rank = pred_reviews[:k].index(true_review) + 1  # Find rank (1-based index)
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)  # True review not in top-K predictions

    return np.mean(reciprocal_ranks)


def prepare_submission_file(accommodation_ids, user_ids, top_k_predictions, review_ids, output_path):
    """
    Prepare the submission file in the required format.
    """
    print("Preparing the submission file...")

    # Map indices to review IDs
    top_k_review_ids = np.array(review_ids)[top_k_predictions]

    # Prepare submission DataFrame
    submission = pd.DataFrame({
        'accommodation_id': accommodation_ids,
        'user_id': user_ids
    })
    for i in range(top_k_review_ids.shape[1]):
        submission[f'review_{i+1}'] = top_k_review_ids[:, i]

    if 'ID' not in submission.columns:
        submission.insert(0, 'ID', range(1, len(submission) + 1))

    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    print("Starting pipeline...")

    # Load and preprocess data
    data = load_and_preprocess_data("../data/train_reviews.csv",
                                    "../data/train_users.csv",
                                    "../data/train_matches.csv")

    # Split data by users
    train_data, test_data = split_data_by_users(data)

    # Prepare train and test features
    X_train_text, X_train_numeric, X_train_categorical, vectorizer = prepare_features(train_data)
    X_test_text, X_test_numeric, X_test_categorical, _ = prepare_features(test_data, vectorizer=vectorizer)

    X_train = combine_features(X_train_text, X_train_numeric, X_train_categorical)
    X_test = combine_features(X_test_text, X_test_numeric, X_test_categorical)

    y_train = train_data['user_id'].values
    y_test = test_data['user_id'].values

    # Train model
    model = train_model(X_train, y_train)

    # Predict probabilities and determine top-10 predictions
    print("Predicting probabilities...")
    y_prob = model.predict_proba(X_test)
    top_k_predictions = np.argsort(y_prob, axis=1)[:, -10:][:, ::-1]  # Reverse for ranking

    # Evaluate model using MRR@10
    mrr_at_10 = calculate_mrr_at_k(y_test, top_k_predictions)
    print(f"MRR@10: {mrr_at_10}")

    # Prepare submission file
    prepare_submission_file(
        accommodation_ids=test_data['accommodation_id'].values,
        user_ids=test_data['user_id'].values,
        top_k_predictions=top_k_predictions,
        review_ids=data['review_id'].values,
        output_path="submission.csv"
    )

    print("Pipeline completed.")
