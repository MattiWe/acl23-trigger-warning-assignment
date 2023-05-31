import click
import json
import logging
from typing import Tuple, List, Iterable
from tqdm import tqdm
from pathlib import Path
from datetime import datetime as dt
from collections import namedtuple, Counter
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
import re

from tokenizers import Tokenizer

import numpy as np
from numpy.typing import ArrayLike

time = dt.now()

tw_fine_open = json.loads(open(Path(__file__).parent / "resources/label-set/tw-fine-open.json").read())
tw_fine_closed = json.loads(open(Path(__file__).parent / "resources/label-set/tw-fine-closed.json").read())
tw_coarse = json.loads(open(Path(__file__).parent / "resources/label-set/tw-coarse.json").read())
closed_labels = set(tw_fine_closed)
open_labels = [_ for _ in tw_fine_open if _ not in closed_labels]

map_fine_to_coarse = json.loads(open(Path(__file__).parent / "resources/label-set/map-fine-to-coarse.json").read())

map_open_to_none = {"other-discrimination": "None", "other-aggression": "None", "other-abuse": "None", "other-pregnancy": "None",
                    "other-medical": "None", "other-mental-health": "None", "other-sexual-content": "None"}

re_punct = re.compile(r'[^\w\s.,!?"\']')
Work = namedtuple('Work', ['work_id', 'freeform', 'support_fine', 'support_coarse', 'coarse', 'fine', 'open', 'closed',
                           'n_words', 'n_chapters'])


def _to_array_representation(labels: List[str], label_set: List[str]) -> Iterable[int]:
    """ convert a string representation of the labels (used in the labels.jsonl)
        into the array representation (used in the works.jsonl).

        The array representation is natively understood by huggingface and scikit-learn.
     """
    return [1 if label in labels else 0 for label in label_set]


def _time(silent=False):
    global time
    now = dt.now()
    if not silent:
        print(f"took {now - time}")
    time = now


def _preprocess(x: str) -> str:
    """ A minimalistic preprocessor: remove all html codes, all non-ascii characters, and lowercase """
    tree = HTMLTree.parse(x)
    x = extract_plain_text(tree, preserve_formatting=False)
    x = x.lower()
    x = re.sub(re_punct, '', x)
    return x


def load_data(dataset: Path) -> Tuple[List, List, ArrayLike]:
    """
    Load a trigger detection dataset and return the id, texts, and labels.
    We do batch-tokenization here (instead of in the vectorizer) so we only have to do it once in the ablation.
    :param dataset: Path to the dataset (file) to load.
    :return: work_id, x, y
    """

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    x_text = []
    y = []
    work_id = []
    for line in tqdm(open(dataset), desc=f"loading data from {dataset}"):
        line = json.loads(line)
        x_text.append(line["text"])
        if 'labels' in line.keys():
            y.append(line["labels"])
        work_id.append(line["work_id"])

    # this removes CLS and SEP token from the BERT tokenizer.
    x_text = tokenizer.encode_batch(x_text)
    x_text = [x.tokens[1:-1] for x in x_text]

    return work_id, x_text, np.asarray(y, dtype=np.int32)


def load_metadata(dataset: Path, tag_map: Path = None) -> List[Work]:
    """ Load metadata of the dataset (i.e. create `Work` objects for each example)

    support = a vector with the number of supporting tags for each warning at the index of tw_fine_open and tw_coarse
    """
    if tag_map:
        ff_map = {}
        for _line in open(tag_map, 'r'):
            tag, warning = _line.strip().split(",")
            ff_map.setdefault(tag.strip(), []).append(warning.strip())
        tag_map = ff_map
    else:
        tag_map = {}

    def _workload(line):
        work = json.loads(line)
        support_fine = []
        support_coarse = []
        for ff in work['freeform']:
            support_fine.extend(tag_map.get(ff, []))
            support_coarse.extend([map_fine_to_coarse[_tag] for _tag in tag_map.get(ff, [])])
        support_fine = Counter(support_fine)
        support_coarse = Counter(support_coarse)

        return Work(work_id=work['work_id'], coarse=work['warnings_coarse_open'], fine=work['warnings_fine_open'],
                    open=[_tag for _tag in work['warnings_fine_open'] if _tag in open_labels],
                    closed=[_tag for _tag in work['warnings_fine_open'] if _tag in closed_labels],
                    support_fine=[support_fine.get(_warning, 0) for _warning in tw_fine_open],
                    support_coarse=[support_coarse.get(_warning, 0) for _warning in tw_coarse],
                    freeform=work['freeform'],
                    n_words=work['n_words'], n_chapters=work['n_chapters'])

    return [_workload(_) for _ in tqdm(open(dataset), desc="load metadata")]


def write_predictions(output_dir: Path, work_ids: List[str], labels: ArrayLike, label_set: List[str]) -> None:
    """
    Write the model predictions to a labels.jsonl
    :param label_set: a list with the labels (in the order user for conversion between list and array style)
    :param output_dir: Path where to write the file to
    :param work_ids: a list of length n with work_ids
    :param labels: a List-like of length n, where each element is a list of labels (array form) and corresponds to the\
                   elment with the same ID in work_ids.
    :return: None
    """
    with open(output_dir / "labels.jsonl", 'w') as of:
        for wid, label_list in zip(work_ids, labels):
            result = {"work_id": wid, "labels": [label_set[idx] for idx, cls in enumerate(label_list) if cls == 1]}
            of.write(f"{json.dumps(result)}\n")


def to_datasets(input_file: Path, output_file: Path, label_set):
    """
    Convert the input_file from the hydrated dataset format to a huggingface (datasets) compatible variant
        with array-style labels and without metadata.
    :param label_set: `warnings_fine_open` or `warnings_coarse_open` --- whatever label-set should be encoded as `labels`
    :param input_file: Path to a .jsonl file with works in the format of the hydrated dataset.
    :param output_file: Path to a .jsonl file to write the output to.
    :return: None.
    """
    ls = tw_fine_open if label_set == 'warnings_fine_open' else tw_coarse
    with open(output_file, 'w') as of:
        for work in tqdm(open(input_file), desc='Converting works ...'):
            work = json.loads(work)
            new_work = {"work_id": work["work_id"],
                        "text": " ".join([_preprocess(_) for _ in work["text"]]),
                        "labels": _to_array_representation(work[label_set], label_set=ls)}
            of.write(f"{json.dumps(new_work)}\n")


if __name__ == "__main__":
    pass
