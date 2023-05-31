import json
import logging
from tqdm import tqdm
import click

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

import numpy as np
from torch import nn
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_auc_score


def _load_dataset(checkpoint: str, train_path: str, validation_path: str, test_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True)

    data_files = {"train": train_path, "test": test_path, "validation": validation_path}
    dataset = load_dataset("json", data_files=data_files)
    dataset.set_format("torch")

    tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=['text'])  # 'work_id'
    tokenized_data = (tokenized_data
                      .map(lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
                      .rename_column("float_labels", "labels"))

    return tokenized_data, tokenizer


def _score(logits):
    threshold = 0.5
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred


def _compute_metric(eval_pred):
    logits, labels = eval_pred
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    y_pred = _score(logits)

    return {'f1': f1_score(y_true=labels, y_pred=y_pred, average='micro'),
            'f1_macro': f1_score(y_true=labels, y_pred=y_pred, average='macro'),
            'accuracy': accuracy_score(labels, y_pred)}


def run_experiment(checkpoint, training, validation, test, savepoint, epochs, batches, lr, name):
    logging.info("load dataset and tokenizer")
    ds, tokenizer = _load_dataset(checkpoint, training, validation, test)

    ds_train = ds["train"]
    ds_validation = ds["validation"]
    ds_test = ds["test"]
    num_labels = len(ds_train[0]["labels"])

    logging.info("start model training")

    args = TrainingArguments(
        savepoint,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batches,
        per_device_eval_batch_size=batches,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        push_to_hub=False,
        report_to="wandb",
        run_name=name)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels,
                                                               problem_type="multi_label_classification").to('cuda')
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_validation,
                      tokenizer=tokenizer, compute_metrics=_compute_metric, resume_from_checkpoint=True)

    trainer.evaluate()
    trainer.train()

    logging.warning("save trained model")
    trainer.save_model(savepoint)

    logging.warning("make predictions")
    predictions = trainer.predict(ds_test)

    results = getattr(predictions, "metrics")
    open(f"{savepoint}/test-results.json", 'w').write(json.dumps(results))

    tokenizer_kwargs = {'padding': True, 'truncation': True}
    classifier = pipeline(model=trainer.model_wrapped, tokenizer=tokenizer, task="text-classification",
                          return_all_scores=True, device="cuda:0")

    def classify(examples):
        p = classifier(examples["text"], **tokenizer_kwargs)
        p = [[1 if elem['score'] > 0.5 else 0 for elem in example] for example in p]
        return {"prediction": p}

    logging.warning("make predictions")
    dataset_with_predictions = ds_test.map(classify, batched=True, batch_size=batches, remove_columns=["text"])

    dataset_with_predictions.to_json(f"{savepoint}/test-predictions.jsonl")


@click.option('-c', '--checkpoint', type=click.STRING, default='allenai/longformer-base-4096',
              help="base checkpoint for model and tokenized")
@click.option('--training', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/train.jsonl",
              help='Path to the training dataset jsonl file')
@click.option('--validation', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/validation.jsonl",
              help='Path to the validation dataset jsonl file')
@click.option('--test', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/test.jsonl",
              help='Path to the test dataset jsonl file')
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the model in.")
@click.option('--epochs', type=click.INT, default=5, help="Path to save the model in.")
@click.option('--batches', type=click.INT, default=4, help="Batch size")
@click.option('--lr', type=click.FLOAT, default=0.00002, help="Initial learning rate")
@click.option('-n', '--name', type=click.STRING, default='develop', help="base name of the model (for wandb)")
@click.command()
def lf_experiment(checkpoint: str, training: str, validation: str, test: str, savepoint: str,
                  epochs: int, batches: int, lr: float, name: str):
    """
    Run the (neural baseline) experiments.

    EP=5
    LR=2e-5
    NAME="lf-28k-2e5lr-5ep-fine"
    INPUT="69k"
    DATA="acl23-coarse-open"

    python3 longformer_experiments.py \
        --training "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/$DATA/train-$INPUT.jsonl" \
        --validation "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/$DATA/validation.jsonl" \
        --test "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/$DATA/test.jsonl" \
        --savepoint "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/$DATA/models/$NAME" \
        --epochs $EP \
        --lr $LR \
        --name $NAME
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_experiment(checkpoint, training, validation, test, savepoint, epochs, batches, lr, name)


if __name__ == "__main__":
    lf_experiment()
