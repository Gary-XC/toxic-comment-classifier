# src/models/train_xlmr.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import Dataset


class ToxicCommentDataset(Dataset):
    """PyTorch Dataset for multi-label abusive language detection."""
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification (Lin et al., 2017).
    Down-weights easy examples and focuses on hard negatives/positives.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Standard BCE With Logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # p_t is the predicted probability for the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal modulation factor
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce
        
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = MultiLabelFocalLoss(gamma=2.0)
    total_loss = 0
    
    # Initialize PyTorch Gradient Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].unsqueeze(1).to(device)
        
        # Cast forward pass to FP16
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
        
        # Scale loss and backpropagate (prevents FP16 gradient underflow)
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the hold-out test set and returns business metrics.
    """
    model.eval()
    criterion = MultiLabelFocalLoss(gamma=2.0)
    total_loss = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Binarize predictions at 0.5 threshold for F1 Score
    preds_binary = (all_preds >= 0.5).astype(int)
    
    metrics = {
        "val_loss": total_loss / len(dataloader),
        "val_pr_auc": average_precision_score(all_labels, all_preds),
        "val_f1_macro": f1_score(all_labels, preds_binary, average="macro")
    }
    return metrics