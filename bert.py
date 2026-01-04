import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nTrain distribution:\n{train_df['sentiment'].value_counts()}")
print(f"\nTest distribution:\n{test_df['sentiment'].value_counts()}")

# Create label mappings
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

train_df['label'] = train_df['sentiment'].map(label2id)
test_df['label'] = test_df['sentiment'].map(label2id)

# Load IndoBERT tokenizer and model
MODEL_NAME = 'indobenchmark/indobert-base-p1'
print(f"\nLoading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,  # Increased dropout for regularization
    attention_probs_dropout_prob=0.3
)

# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("\nCreating datasets...")
train_dataset = SentimentDataset(
    train_df['text_clean'].values,
    train_df['label'].values,
    tokenizer
)

test_dataset = SentimentDataset(
    test_df['text_clean'].values,
    test_df['label'].values,
    tokenizer
)

# Compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments - optimized for small dataset
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # More epochs with early stopping
    per_device_train_batch_size=8,  # Small batch size
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,  # Small learning rate
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    fp16=False,  # Set to True if using CUDA
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("\nStarting training...")
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate()

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
print(f"Test F1-Score: {test_results['eval_f1']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test Recall: {test_results['eval_recall']:.4f}")

# Get detailed predictions for classification report
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(
    labels, 
    preds, 
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))

# Show some example predictions
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)

for i in range(min(5, len(test_df))):
    text = test_df.iloc[i]['text_clean']
    actual = id2label[labels[i]]
    predicted = id2label[preds[i]]
    
    print(f"\nText: {text[:80]}...")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print(f"Match: {'✓' if actual == predicted else '✗'}")

# Save the model
print("\nSaving model...")
model.save_pretrained('./indobert_sentiment_model')
tokenizer.save_pretrained('./indobert_sentiment_model')
print("Model saved to ./indobert_sentiment_model")

# Export results to JSON
print("\nExporting results to JSON...")

# Calculate confusion matrix
cm = confusion_matrix(labels, preds)

# Per-class detailed metrics
from sklearn.metrics import precision_recall_fscore_support
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    labels, preds, average=None, zero_division=0
)

# Prepare example predictions
example_predictions = []
for i in range(min(10, len(test_df))):
    example_predictions.append({
        'text': test_df.iloc[i]['text_clean'],
        'actual': id2label[labels[i]],
        'predicted': id2label[preds[i]],
        'correct': bool(labels[i] == preds[i])
    })

results_json = {
    'model_info': {
        'model_name': MODEL_NAME,
        'model_type': 'IndoBERT',
        'num_labels': 3,
        'labels': ['negative', 'neutral', 'positive'],
        'timestamp': datetime.now().isoformat()
    },
    'dataset_info': {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_distribution': train_df['sentiment'].value_counts().to_dict(),
        'test_distribution': test_df['sentiment'].value_counts().to_dict()
    },
    'overall_metrics': {
        'accuracy': float(test_results['eval_accuracy']),
        'f1_score': float(test_results['eval_f1']),
        'precision': float(test_results['eval_precision']),
        'recall': float(test_results['eval_recall'])
    },
    'per_class_metrics': {
        'negative': {
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0]),
            'f1_score': float(f1_per_class[0]),
            'support': int(support_per_class[0])
        },
        'neutral': {
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1]),
            'f1_score': float(f1_per_class[1]),
            'support': int(support_per_class[1])
        },
        'positive': {
            'precision': float(precision_per_class[2]),
            'recall': float(recall_per_class[2]),
            'f1_score': float(f1_per_class[2]),
            'support': int(support_per_class[2])
        }
    },
    'confusion_matrix': {
        'matrix': cm.tolist(),
        'labels': ['negative', 'neutral', 'positive']
    },
    'example_predictions': example_predictions
}

with open('bert_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

print("Results saved to bert_results.json")
print("\nDone!")
