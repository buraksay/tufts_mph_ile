import optuna
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb
import os
import argparse
import json




def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    
    # Convert logits to probabilities using PyTorch's softmax
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    # Compute AUCROC
    if probs.shape[1] == 2:  # Binary classification
        aucroc = roc_auc_score(labels, probs[:, 1])
    else:  # Multi-class classification
        aucroc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'aucroc': aucroc
    }


def make_dataset():
    RES_DIR = os.environ.get("RES_DIR",
        os.path.join("cluster", "home", "bsay01", "aphaproject", "results"))

    data_dir = os.path.join(RES_DIR, 'finebert_seed202410_spliton2020')

    x_train = load_dataset('csv', data_files=os.path.join(data_dir, 'x_train.csv'))['train']
    y_train = load_dataset('csv', data_files=os.path.join(data_dir, 'y_train.csv'))['train']
    x_valid = load_dataset('csv', data_files=os.path.join(data_dir, 'x_valid.csv'))['train']
    y_valid = load_dataset('csv', data_files=os.path.join(data_dir, 'y_valid.csv'))['train']
    x_test = load_dataset('csv', data_files=os.path.join(data_dir, 'x_test.csv'))['train']
    y_test = load_dataset('csv', data_files=os.path.join(data_dir, 'y_test.csv'))['train']

    train_dataset = x_train.add_column('IsORI', y_train['IsORI'])
    valid_dataset = x_valid.add_column('IsORI', y_valid['IsORI'])
    test_dataset = x_test.add_column('IsORI', y_test['IsORI'])

    # Create the final dataset
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })

    # Set 'IsORI' as the label column
    dataset = dataset.rename_column('IsORI', 'label')
    return dataset

def model_init():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def run_trial(trial_number):
    # Initialize wandb
    #wandb.login()
    # Initialize wandb
    # wandb.login(key=os.environ.get("WANDB_API_KEY"))
    # wandb.login(key="bd54b12223aa5e3c4cfc59aaf9a15dfb7084ce51")
    wandb.login(
        key=os.environ.get("WANDB_API_KEY", "bd54b12223aa5e3c4cfc59aaf9a15dfb7084ce51")
    )

    # Load your dataset
    dataset = make_dataset()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["Narrative"], padding="max_length", truncation=True)   
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load or create the Optuna study
    study_name = "bert-finetuning-study-temporal-long"
    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")

    # Get the trial object for this run
    trial = study.ask()

    # Define hyperparameters for this trial
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    num_train_epochs = trial.suggest_int('num_train_epochs', 15, 200)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    # Initialize a new wandb run
    with wandb.init(project=study_name, config=trial.params, name=f"trial-{trial_number}"):
        training_args = TrainingArguments(
            output_dir=f"./results/trial_{trial_number}",
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb",
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()

    # predict on test dataset
    # test_dataset = tokenized_datasets["test"]
    predictions = trainer.predict(tokenized_datasets["test"])
    result = compute_metrics(predictions)

    # Report the result to Optuna
    study.tell(trial, eval_result["eval_accuracy"])

    # Save the trial results
    os.makedirs("trial_results", exist_ok=True)
    with open(f"trial_results/trial_{trial_number}.json", "w") as f:
        json.dump({
            "trial_number": trial_number,
            "params": trial.params,
            "value": result["aucroc"]
        }, f)

    print(f"Trial {trial_number} completed. Test aucroc: {result['aucroc']}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='BERT fine-tuning with hyperparameter search')
    parser.add_argument('--trial', type=int, required=True, help='Trial number for this run')
    args = parser.parse_args()

    # Run the trial
    run_trial(args.trial)
