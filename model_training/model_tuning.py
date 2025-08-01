from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

from functions import *

def load_data(csv_path='../data/dummy_job_fit_data.csv'):
    return pd.read_csv(csv_path)

def preprocess_features(df):
    feature_cols = ['skill_overlap', 'edu_match', 'exp_gap', 'candidate_edu', 'job_req_edu']
    X = df[feature_cols]
    y = df['fit_label']
    return X, y

def build_pipeline():
    numeric_features = ['skill_overlap', 'edu_match', 'exp_gap']
    categorical_features = ['candidate_edu', 'job_req_edu']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=2000))
    ])

    return pipeline

def main():
    df = load_data()
    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()

    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['lbfgs', 'saga']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)

    report_text = f"Tuned model: \n\n Accuracy: {acc:.4f}\n\nClassification Report:\n{report} \n\n Best params: {grid_search.best_params_}"
    results_png_path = os.path.join(results_dir, 'tuned_model_results.png')
    save_text_as_image(report_text, results_png_path)


if __name__ == '__main__':
    main()
