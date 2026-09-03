import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.special import softmax

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_C_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector"
RESULTS_DIR = TIER_C_DIR / "results"

MODEL_NAME = "roberta-base"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = softmax(pred.predictions, axis=1)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0
        
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc)
    }

def train_experiment(exp_name, train_df, val_df, test_df):
    print(f"\\n{'='*50}\\nStarting {exp_name}\\n{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Convert to HuggingFace Datasets
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "paragraph_id", "author", "book", "topic"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    # Load Model & Apply LoRA
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    output_dir = MODELS_DIR / exp_name
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(RESULTS_DIR / "logs" / exp_name),
        logging_steps=10,
        seed=42
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Training...")
    trainer.train()
    
    print("Evaluating on Test Set...")
    test_results = trainer.predict(tokenized_datasets["test"])
    
    # Save the adapter
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    return test_results.metrics, test_results.predictions, test_results.label_ids, test_df

def main():
    df = pd.read_csv(DATA_CSV)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    
    # --- EXPERIMENT 1: Random Split (80/10/10) ---
    train_val_df, test_df_r = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    train_df_r, val_df_r = train_test_split(train_val_df, test_size=0.1111, stratify=train_val_df['label'], random_state=42)
    
    metrics_r, preds_r, labels_r, test_data_r = train_experiment("random_split", train_df_r, val_df_r, test_df_r)
    all_metrics["random_split"] = metrics_r
    
    # --- EXPERIMENT 2: Book-level Split ---
    train_books = ['oliver_twist', 'great_expectations', 'emma', 'sense_and_sensibility']
    test_books = ['pride_and_prejudice', 'a_tale_of_two_cities']
    
    train_val_df_b = df[df['book'].isin(train_books)]
    test_df_b = df[df['book'].isin(test_books)]
    
    # Take 10% of training data as validation
    train_df_b, val_df_b = train_test_split(train_val_df_b, test_size=0.1, stratify=train_val_df_b['label'], random_state=42)
    
    metrics_b, preds_b, labels_b, test_data_b = train_experiment("book_split", train_df_b, val_df_b, test_df_b)
    all_metrics["book_split"] = metrics_b
    
    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print("Metrics saved to results/metrics.json")
    
if __name__ == "__main__":
    main()
