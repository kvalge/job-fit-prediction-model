import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

from functions import *


def load_data(csv_path='data/dummy_job_fit_data.csv'):
    return pd.read_csv(csv_path)

def preprocess_and_train(df):
    feature_cols = ['skill_overlap', 'edu_match', 'exp_gap', 'candidate_edu', 'job_req_edu']
    X = df[feature_cols]
    y = df['fit_label']

    categorical_features = ['candidate_edu', 'job_req_edu']
    numeric_features = ['skill_overlap', 'edu_match', 'exp_gap']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)

    report_text = f"Preliminary model: \n\n Accuracy: {acc:.4f}\n\nClassification Report:\n{report}"
    results_png_path = os.path.join(results_dir, 'preliminary_model_results.png')
    save_text_as_image(report_text, results_png_path)

    return clf


if __name__ == '__main__':
    csv_file = '../data/dummy_job_fit_data.csv'
    df = load_data(csv_file)
    model = preprocess_and_train(df)
