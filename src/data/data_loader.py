# src/data/dataset.py
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class ToxicCommentDataset(Dataset):
    """
    PyTorch Dataset for multi-label abusive language detection.
    Expects text strings and binary/float labels.
    """
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer: PreTrainedTokenizer, max_len: int = 128):
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # We use tokenizer() directly which yields dictionary of tensors
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(), # type: ignore
            'attention_mask': encoding['attention_mask'].flatten(), # type: ignore
            'labels': label
        }

def load_and_filter_jigsaw_data(
    csv_path: str, 
    target_langs: list[str] | None = None,
    text_col: str = 'comment_text',
    target_col: str = 'target',
    lang_col: str = 'lang',
    threshold: float = 0.5,
    sample_size: int = 100000
) -> pd.DataFrame:
    """
    Loads Jigsaw Unintended Bias data, binarizes the target, and filters languages.
    """
    if target_langs is None:
        target_langs = ['en', 'fr', 'es', 'zh']
    logger.info(f"Loading dataset from {csv_path}...")
    # Use chunking or pyarrow backend for large CSVs to prevent OOM errors
    df = pd.read_csv(csv_path, engine="c", low_memory=False)
    
    if target_col not in df.columns:
        if 'toxic' in df.columns:
            logger.info(f"Column '{target_col}' not found, but 'toxic' exists. Normalizing schema.")
            df.rename(columns={'toxic': target_col}, inplace=True)
        else:
            raise KeyError(
                f"Label column missing. Expected '{target_col}' or 'toxic'. "
                f"Found columns: {df.columns.tolist()}"
            )
    
    # Binarize continuous target to 1 (Toxic) and 0 (Benign)
    df['toxic_binary'] = (df[target_col] >= threshold).astype(int)
    
    if lang_col in df.columns:
        df = df[df[lang_col].isin(target_langs)]
        logger.info(f"Filtered to target languages. Remaining distribution:\n{df[lang_col].value_counts()}")
        
        if 'zh' not in df[lang_col].unique():
            logger.warning("Chinese ('zh') not found in dataset. Relying on XLM-R Zero-Shot Transfer for Mandarin.")
    else:
        logger.warning(f"No '{lang_col}' column found. Assuming all data is English.")
    
    # Handling Class Imbalance: Downsample the benign class (majority) to speed up 
    df_toxic = df[df['toxic_binary'] == 1]
    df_benign = df[df['toxic_binary'] == 0].sample(n=len(df_toxic) * 2, random_state=42) # 2:1 ratio
    
    df_balanced = pd.concat([df_toxic, df_benign]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Limit sample size for faster iteration during portfolio development
    if len(df_balanced) > sample_size:
        df_balanced = df_balanced.head(sample_size)
        
    return df_balanced