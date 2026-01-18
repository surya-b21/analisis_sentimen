import pandas as pd
import numpy as np
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def main():
    print("Loading data...")
    df = pd.read_csv('dataset_used.csv', encoding='utf-8', on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()  # Remove whitespace from column names

    # label mapping
    label2id = {'negative': 0, 'positive': 1}
    id2label = {0: 'negative', 1: 'positive'}

    df['label'] = df['label'].str.strip().map(label2id)  # Remove whitespace and map labels

    # membagi data menjadi 80% train dan 20% test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # konversi teks ke dalam bentuk Dataset
    train_dataset = Dataset.from_pandas(train_df[['comment', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['comment', 'label']])

    # load model
    MODEL_NAME = 'indobenchmark/indobert-base-p1'
    print(f"\nLoading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # apply tokenization
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x['comment'], padding='max_length', truncation=True, max_length=128),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenizer(x['comment'], padding='max_length', truncation=True, max_length=128),
        batched=True
    )

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # train model
    print("\nTraining model...")
    trainer.train()

    # evaluate model
    print("\nEvaluating model...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    # predict on test set
    print("\nMaking predictions on test set...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        true_labels, 
        pred_labels, 
        target_names=['negative', 'positive']
    ))

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    
    labels_list = ['negative', 'positive']
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f'Actual {label}' for label in labels_list],
        columns=[f'Predicted {label}' for label in labels_list]
    )
    print(conf_df)

    # Prepare results for JSON export
    per_class_metrics = {}
    for idx, label in enumerate(labels_list):
        per_class_metrics[label] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx])
        }

    results = {
        'model': 'IndoBERT',
        'model_name': 'indobenchmark/indobert-base-p1',
        'overall_metrics': {
            'accuracy': float(accuracy),
            'accuracy_percentage': float(accuracy * 100)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'matrix': conf_matrix.tolist(),
            'labels': labels_list
        },
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'classes': labels_list
        },
        'training_config': {
            'num_epochs': 3,
            'batch_size': 8,
            'max_length': 128,
            'learning_rate': 2e-5
        }
    }

    # Save to JSON
    with open('indobert_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Results saved to indobert_results.json")

if __name__ == "__main__":
    main()