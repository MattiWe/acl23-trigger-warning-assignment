import logging
from tqdm import tqdm
from typing import Set, List, Dict, Tuple, Any, Union, Iterable
import json
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


x_labels = {
    "open": {
        "coarse": ["discrimination", "aggression", "pregnancy", "anatomy", "death", "mental-health", "sexual"],
        "fine": ["pornographic-content", "other-sexual", "violence", "other-mental-health", "death", "other-aggression",
                 "sexual-assault", "abuse", "blood", "other-death", "suicide", "pregnancy",  "child-abuse", "incest",
                 "underage", "homophobia", "self-harm", "dying", "kidnapping", "other-anatomy", "mental-illness",
                 "dissection", "eating-disorders", "abduction",  "body-hatred", "other-discrimination", "childbirth",
                 "racism", "sexism", "miscarriages", "transphobia", "abortion", "fat-phobia", "animal-death",
                 "ableism", "classism", "other-pregnancy", "misogyny",  "animal-cruelty", "religious-discrimination",
                 "heterosexism"]
    },
    "closed": {
        "coarse": ["discrimination", "aggression", "pregnancy", "anatomy", "death", "mental-health", "sexual"],
        "fine": ["pornographic-content", "violence", "death", "sexual-assault", "abuse", "blood", "suicide",
                 "pregnancy",  "child-abuse", "incest", "underage", "homophobia", "self-harm", "dying", "kidnapping",
                 "mental-illness", "dissection", "eating-disorders", "abduction",  "body-hatred", "childbirth",
                 "racism", "sexism", "miscarriages", "transphobia", "abortion", "fat-phobia", "animal-death",
                 "ableism", "classism", "misogyny",  "animal-cruelty", "religious-discrimination", "heterosexism"]
}}


def _evaluate(y_true, y_predictions):
    return {
        "micro": {
            "precision": precision_score(y_true, y_predictions, average='micro'),
            "recall": recall_score(y_true, y_predictions, average='micro'),
            "f1": f1_score(y_true, y_predictions, average='micro'),
        },
        "macro": {
            "precision": precision_score(y_true, y_predictions, average='macro'),
            "recall": recall_score(y_true, y_predictions, average='macro'),
            "f1": f1_score(y_true, y_predictions, average='macro'),
        },
        "accuracy": accuracy_score(y_true, y_predictions)
    }

def compile_hf_datasets(dataset_path: Path, ml_dataset_path: Path, label_names: Iterable,
                        crop: Union[int, None] = None, work_types: Union[str, None] = None):
    """ this builds derivative datasets that can be used by HF datasets
        Removes all works with more than 1 chapter and more than `crop` tokens.
        Adds the labels als dense vector

        This was used to train the longformer
     """

    ml_dataset_path.mkdir(exist_ok=True)
    if not work_types:
        work_types = ['training', 'validation', 'test']

    for work_type in work_types:
        fic_file = list(dataset_path.glob(f"*{work_type}*-fics.jsonl"))[0]
        label_file = list(dataset_path.glob(f"*{work_type}*-labels.jsonl"))[0]
        labels = {json.loads(line)["work_id"]: set(json.loads(line)["labels"]) for line in open(label_file)}
        (ml_dataset_path / dataset_path.stem).mkdir(exist_ok=True)
        with open(ml_dataset_path / dataset_path.stem / f"{work_type}.jsonl", 'w') as of:
            for line in tqdm(open(fic_file), desc="fics"):
                line = json.loads(line)
                if len(line["text"]) > 1:
                    continue
                if crop and len(line["text"][0].split(" ")) > crop:
                    continue
                text = line["text"][0]
                lab = [1 if label in labels[line["work_id"]] else 0 for label in label_names]
                new_line = {"idx": line["work_id"], "text": text, "labels": lab}
                of.write(f"{json.dumps(new_line)}\n")


def compile_pretokenized(dataset_directories, ml_dataset_path):
    """ This tokenizes the datasets using the longformer tokenizer (so the features can be build quickly)

    The result was used to train the SVM and XGBoost datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
    ml_dataset_path.mkdir(exist_ok=True)
    for dataset_path in dataset_directories:
        openness = dataset_path.stem.split("-")[0]
        granularity = dataset_path.stem.split("-")[2]
        for work_type in ['training', 'validation', 'test']:
            fic_file = list(dataset_path.glob(f"*{work_type}*-fics.jsonl"))[0]
            label_file = list(dataset_path.glob(f"*{work_type}*-labels.jsonl"))[0]
            labels = {json.loads(line)["work_id"]: set(json.loads(line)["labels"]) for line in open(label_file)}
            (ml_dataset_path / dataset_path.stem).mkdir(exist_ok=True)
            with open(ml_dataset_path / dataset_path.stem / f"{work_type}.jsonl", 'w') as of:
                for line in tqdm(open(fic_file), desc="fics"):
                    line = json.loads(line)
                    text = []
                    for chapter in tokenizer(line["text"])["input_ids"]:
                        text.extend(chapter)
                    lab = [1 if label in labels[line["work_id"]] else 0
                           for label in x_labels[openness][granularity]]
                    new_line = {"idx": line["work_id"], "text": text, "labels": lab}
                    of.write(f"{json.dumps(new_line)}\n")


def main():
    base_path = Path("..")
    dataset_dir = [base_path / "data" / "open-set-coarse-grained",
                   base_path / "data" / "open-set-fine-grained",
                   base_path / "data" / "closed-set-coarse-grained",
                   base_path / "data" / "closed-set-fine-grained"]

    for dataset_path in dataset_dir:
        openness = dataset_path.stem.split("-")[0]
        granularity = dataset_path.stem.split("-")[2]
        compile_hf_datasets(dataset_path, base_path / "data" / "longformer-training", crop=4000,
                            label_names=x_labels[openness][granularity])

    compile_pretokenized(dataset_dir, base_path / "dataset" / "svm")


if __name__ == "__main__":
    main()
