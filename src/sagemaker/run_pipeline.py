# src/sagemaker/run_pipeline.py
import argparse

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast

from src.data.data_loader import ToxicCommentDataset, load_and_filter_jigsaw_data

# Importing the models
from src.models.baseline import train_baseline
from src.models.train_xlmr import evaluate_model, train_epoch


def run_pipeline(data_path: str):
    """
    End-to-End orchestrator. Compares Baseline vs XLM-R and logs to MLflow.
    """
    # 1. Setup MLflow Tracking (Local SQLite for now, points to S3/RDS in production)
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Multilingual_Toxicity_Detection")
    
    # 2. Load and Prepare Data
    df = load_and_filter_jigsaw_data(data_path, sample_size=2000)
    df_train, df_test = train_test_split(df, test_size=0.15, stratify=df['toxic_binary'], random_state=42)
    
    # =====================================================================
    # STAGE 1: Train & Log the Baseline (Logistic Regression)
    # =====================================================================
    with mlflow.start_run(run_name="Char_TFIDF_LogReg_Baseline"):
        print("--- Training Baseline ---")
        # Log algorithmic parameters to establish the scientific baseline
        mlflow.log_params({"model_type": "LogisticRegression", "features": "Char-TFIDF", "ngrams": "(1,4)"})
        
        metrics = train_baseline(df_train, df_test, text_col='comment_text', label_col='toxic_binary')
        mlflow.log_metrics(metrics)
        
        # Note: We rely on the baseline script to save the joblib file, 
        # then we log it as an artifact so it is versioned alongside the run.
        mlflow.log_artifact("data/baseline_model.joblib")
        
    # =====================================================================
    # STAGE 2: Train & Log the Neural Architecture (XLM-RoBERTa)
    # =====================================================================
    with mlflow.start_run(run_name="XLM_RoBERTa_FocalLoss"):
        print("\n--- Training XLM-R ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_name = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device) # type: ignore
        
        # DataLoaders for both Train AND Test
        train_dataset = ToxicCommentDataset(df_train['comment_text'], df_train['toxic_binary'], tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        
        test_dataset = ToxicCommentDataset(df_test['comment_text'], df_test['toxic_binary'], tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        
        epochs = 3
        mlflow.log_params({
            "model_type": model_name,
            "loss_function": "Focal_Loss_gamma_2.0",
            "epochs": epochs,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "zero_shot_languages": "zh"
        })
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
        print("\nEvaluating XLM-R on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
        mlflow.log_metrics(test_metrics)
        print(f"Test Metrics: {test_metrics}")
        
        components = {"model": model, "tokenizer": tokenizer}  # noqa: F841
        mlflow.pytorch.log_model( # type: ignore
            model, 
            artifact_path="model_artifacts",
            serialization_format="pickle"
        )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to jigsaw CSV")
    args = parser.parse_args()
    
    run_pipeline(args.data_path)