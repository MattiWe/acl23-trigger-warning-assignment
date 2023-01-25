import json
import logging
from tqdm import tqdm
import click
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_auc_score

from util import x_labels, _evaluate


def _load_dataset(dataset_path, checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True)

    data_files = {"train": str((dataset_path / "training.jsonl").resolve()),
                  "test": str((dataset_path / "test.jsonl").resolve()),
                  "validation": str((dataset_path / "validation.jsonl").resolve())}
    dataset = load_dataset("json", data_files=data_files)
    dataset.set_format("torch")

    tokenized_data = dataset.map(tokenize_function, batched=True)
    tokenized_data = (tokenized_data
                      .map(lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels", 'idx', 'text'])
                      .rename_column("float_labels", "labels"))

    return tokenized_data, tokenizer


def __multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def _compute_metric(p):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = __multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def __model_init(checkpoint, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels,
                                                              problem_type="multi_label_classification").to('cuda')


def _train_model(dataset, tokenizer, checkpoint, name, savepoint: str, batch_size=4, epochs=3):
    ds_train = dataset["train"]
    ds_validation = dataset["train"]
    num_labels = len(ds_train[0]["labels"])

    # TODO https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
    args = TrainingArguments(
        savepoint + f"/longformer-finetuned-{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        push_to_hub=False)

    model = __model_init(checkpoint, num_labels)
    trainer = Trainer(model=model,
                      args=args, train_dataset=ds_train, eval_dataset=ds_validation,
                      tokenizer=tokenizer, compute_metrics=_compute_metric)

    trainer.evaluate()  # only returns the loss?
    trainer.train()
    return trainer
    # TODO https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=NboJ7kDOIrJq
    # best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")


def run_experiment(dataset_path: str, checkpoint: str, savepoint: str, epochs=3):
    logging.warning("load dataset and tokenizer")
    dataset_path = Path(dataset_path)
    ds, tok = _load_dataset(dataset_path, checkpoint)
    name = f"longformer-{dataset_path.stem}"
    logging.warning("start model training")
    trainer = _train_model(ds, tok, checkpoint, name, savepoint, epochs=epochs)
    logging.warning("save trained model")
    trainer.save_model(savepoint)

    return savepoint


def make_prediction(eval_dir: str, model_checkpoint: str, tok_checkpoint: str, open_or_closed: str, coarse_or_fine: str):
    # 2. load model
    logging.warning("load model and tokenizer")
    test_file = Path(eval_dir) / "test-labels.jsonl"
    fics_file = Path(eval_dir) / "test-fics.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                               problem_type="multi_label_classification").to('cuda')

    def _predict_labels(text):
        encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.to(model.device) for k, v in encoding.items()}
        pt_outputs = model(**encoding)
        logits = pt_outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [x_labels[open_or_closed][coarse_or_fine][idx] for idx, label in enumerate(predictions) if
                            label == 1.0]
        return predicted_labels

    logging.warning("start predictions")
    predictions = {}
    with open(model_checkpoint + "/predictions.jsonl", 'w') as of:
        for line in tqdm(open(fics_file), desc='iterating examples'):
            line = json.loads(line)
            labels = {label for chapter in line["text"] for label in _predict_labels(chapter)}
            p = {"work_id": line["work_id"], "labels": list(labels)}
            predictions[line["work_id"]] = list(labels)
            of.write(f"{json.dumps(p)}\n")

    logging.warning("evaluate predictions")
    # evaluate
    Y_pred = []
    y_test = []
    def _binarize(label_list):
        return [1 if label in label_list else 0
                for label in x_labels[open_or_closed][coarse_or_fine]]

    for line in open(test_file):
        line = json.loads(line)
        Y_pred.append(_binarize(predictions[line["work_id"]]))
        y_test.append(_binarize(line["labels"]))

    eval = _evaluate(y_test, Y_pred)
    logging.info(eval)
    open(model_checkpoint + "/evaluation.json", 'w').write(json.dumps(eval))


@click.group()
def cli():
    pass


@click.option('-c', '--checkpoint', type=click.STRING, default='allenai/longformer-base-4096', help="base checkpoint for model and tokenized")
@click.option('-d', '--dataset-dir', type=click.Path(exists=True), help="Path where the training.jsonl and validation.jsonl is.")
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the model in.")
@click.option('--epochs', type=click.INT, default=3, help="Path to save the model in.")
@cli.command()
def train(checkpoint, dataset_dir, savepoint, epochs):
    """
    $ python3 longformer_experiments.py train \
        -d <dataset> \
        -s ./models
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_experiment(dataset_dir, checkpoint, savepoint, epochs=epochs)
    logging.info(f"trained model at {savepoint}")


@click.option('-c', '--checkpoint', type=click.STRING, default='allenai/longformer-base-4096',
              help="base checkpoint for model and tokenized")
@click.option('-t', '--test-file', type=click.Path(exists=True), help="Path to a *-test-labels.jsonl.")
@click.option('-s', '--savepoint', type=click.Path(exists=True), help="Path to save the model in.")
@click.option('-o', '--open-or-closed', type=click.STRING, default='open', help="open or closed as string")
@click.option('-f', '--coarse-or-fine', type=click.STRING, default='fine', help="coarse or fine as string")
@cli.command()
def predict(checkpoint, test_file, savepoint, open_or_closed, coarse_or_fine):
    """
    $ python3 longformer_experiments.py predict \
        -t <dataset>/test-labels.jsonl \
        -s ./models \
        -o "open" -f "fine"
    """
    make_prediction(test_file,
                    model_checkpoint=savepoint,
                    tok_checkpoint=checkpoint,
                    open_or_closed=open_or_closed, coarse_or_fine=coarse_or_fine)


if __name__ == "__main__":
    cli()
