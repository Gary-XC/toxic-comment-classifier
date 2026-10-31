# src/models/baseline.py

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline


def train_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame, text_col: str, label_col: str) -> dict[str, float]:
    """
    Trains a character-level TF-IDF Logistic Regression baseline.
    """
    # Pipeline: Char N-Grams natively handle Mandarin text and Latin typos
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=25000)),
        # class_weight='balanced' inversely scales weights by class frequencies
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs'))
    ])
    
    print("Training baseline model...")
    pipeline.fit(df_train[text_col], df_train[label_col])
    
    # Validation
    y_pred_proba = pipeline.predict_proba(df_test[text_col])[:, 1]
    y_pred = pipeline.predict(df_test[text_col])
    
    metrics = {
        "pr_auc": average_precision_score(df_test[label_col], y_pred_proba),
        "f1_macro": f1_score(df_test[label_col], y_pred, average="macro")
    }
    
    joblib.dump(pipeline, "data/baseline_model.joblib")
    return metrics