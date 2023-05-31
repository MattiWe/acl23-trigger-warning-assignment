import json
import logging
from tqdm import tqdm
from typing import Tuple, List, Dict, Callable
import click
import numpy as np
from functools import partial

from pathlib import Path
from trigger_multilabel_classification.util import load_metadata, Work, tw_fine_open, tw_coarse, \
    open_labels, closed_labels

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


warnings_ordered_by_frequency_fine = ['pornography', 'violence', 'other-mental-health', 'death', 'other-sexual',
                                      'sexual-abuse', 'abuse', 'other-medical', 'blood-gore', 'abusive-language',
                                      'suicide', 'child-abuse', 'childbirth', 'mental-illness', 'addiction', 'incest',
                                      'homophobia', 'self-harm', 'kidnapping', 'other-aggression',
                                      'collective-violence', 'procedures', 'dysmorphia', 'other-pregnancy',
                                      'other-abuse', 'sexism', 'other-discrimination', 'racism', 'miscarriage',
                                      'animal-abuse', 'transphobia', 'abortion', 'ableism',
                                      'religious-discrimination', 'classism', 'body-shaming']


warnings_ordered_by_frequency_coarse = ['sexual-content', 'aggression', 'abuse', 'mental-health', 'medical',
                                        'pregnancy', 'discrimination']

section_indices = {
    "top": {33, 9, 32, 11, 35, 15, 14, 26, 24, 18, 29, 16},
    "mid": {22, 30, 31, 34, 0, 28, 10, 13, 12, 25, 27, 23},
    "bot": {19, 3, 8, 4, 20, 17, 1, 21, 6, 5, 2, 7}
}


def __score(y_true: List[List[int]], y_predictions: List[List[int]], extended: bool,
            label_set: List = tw_fine_open) -> dict:
    y_true = np.asarray(y_true)
    y_predictions = np.asarray(y_predictions)
    results = {
        "mac_f1": round(f1_score(y_true, y_predictions, average='macro', zero_division=0), 2),
        "mac_p": round(precision_score(y_true, y_predictions, average='macro', zero_division=0), 2),
        "mac_r": round(recall_score(y_true, y_predictions, average='macro', zero_division=0), 2),
        "mic_f1": round(f1_score(y_true, y_predictions, average='micro', zero_division=0), 2),
        "mic_p": round(precision_score(y_true, y_predictions, average='micro', zero_division=0), 2),
        "mic_r": round(recall_score(y_true, y_predictions, average='micro', zero_division=0), 2),
        "sub_acc": round(accuracy_score(y_true, y_predictions), 2)
    }
    if extended:
        results["roc_auc"] = round(roc_auc_score(y_true, y_predictions), 2)

        if label_set == tw_fine_open:
            top_true = [value for idx, value in enumerate(y_true) if idx in section_indices["top"]]
            top_predictions = [value for idx, value in enumerate(y_predictions) if idx in section_indices["top"]]
            mid_true = [value for idx, value in enumerate(y_true) if idx in section_indices["mid"]]
            mid_predictions = [value for idx, value in enumerate(y_predictions) if idx in section_indices["mid"]]
            bot_true = [value for idx, value in enumerate(y_true) if idx in section_indices["bot"]]
            bot_predictions = [value for idx, value in enumerate(y_predictions) if idx in section_indices["bot"]]
            results["sections"] = {}
            results["sections"]["top_f1"] = round(f1_score(top_true, top_predictions, average='micro', zero_division=0), 2)
            results["sections"]["top_p"] = round(precision_score(top_true, top_predictions, average='micro', zero_division=0), 2)
            results["sections"]["top_r"] = round(recall_score(top_true, top_predictions, average='micro', zero_division=0), 2)
            results["sections"]["mid_f1"] = round(f1_score(mid_true, mid_predictions, average='micro', zero_division=0), 2)
            results["sections"]["mid_p"] = round(precision_score(mid_true, mid_predictions, average='micro', zero_division=0), 2)
            results["sections"]["mid_r"] = round(recall_score(mid_true, mid_predictions, average='micro', zero_division=0), 2)
            results["sections"]["bot_f1"] = round(f1_score(bot_true, bot_predictions, average='micro', zero_division=0), 2)
            results["sections"]["bot_p"] = round(precision_score(bot_true, bot_predictions, average='micro', zero_division=0), 2)
            results["sections"]["bot_r"] = round(recall_score(bot_true, bot_predictions, average='micro', zero_division=0), 2)

        results["classes"] = {}
        for idx, label in enumerate(label_set):
            results["classes"].setdefault(label, {})[f"p"] = round(
                precision_score(y_true[:, idx], y_predictions[:, idx], average='micro', labels=[1], zero_division=0), 2)
            results["classes"][label][f"r"] = round(
                recall_score(y_true[:, idx], y_predictions[:, idx], average='micro', labels=[1], zero_division=0), 2)
            results["classes"][label][f"f1"] = round(f1_score(y_true[:, idx], y_predictions[:, idx], average='micro',
                                                     labels=[1], zero_division=0), 2)

    return results


def __filter(_file: Path, metadata: Dict[str, Work] = None, length: (int, int) = None,
             min_support: int = None):
    """ Load truth and predictions from the file.
        Filter works by criteria if the criteria are not None

    :param _file: Path to the results file
    :param metadata: metadata dict with {work_id: Work}
    :param length: allowed range of word counts
    :return: a tuple (truth, prediction). Each is a {0, 1}-vector
    """
    logger.debug(f"filter by length {length} and min support {min_support}")
    for line in open(_file):
        line = json.loads(line)
        if length and (metadata.get(line["work_id"]).n_words < length[0]
                       or metadata.get(line["work_id"]).n_words > length[1]):
            continue
        if min_support and 'coarse' in _file.stem:
            if [1 for elem in metadata.get(line["work_id"]).support_coarse if 0 < elem < min_support]:
                continue
        if min_support and 'fine' in _file.stem:
            if [1 for elem in metadata.get(line["work_id"]).support_fine if 0 < elem < min_support]:
                continue

        if "prediction" in line:
            yield line["labels"], line.get("prediction")
        else:
            yield line["truth"], line.get("predictions")


def __filter_openness(_file: Path, return_open: bool = True):
    """ only return the open/closed indices """
    logger.debug(f"filter by openness: open = {return_open}")
    if return_open:
        indices = set([idx for idx, label in enumerate(tw_fine_open) if label in open_labels])
    else:
        indices = set([idx for idx, label in enumerate(tw_fine_open) if label in closed_labels])
    for truth, prediction in __filter(_file=_file):
        truth = [v for idx, v in enumerate(truth) if idx in indices]
        prediction = [v for idx, v in enumerate(prediction) if idx in indices]
        yield truth, prediction


def _evaluate(results_file: Path, label_set: List[str], _filter: Callable = __filter, extended=False):
    predictions = []
    truth = []
    for t, p in _filter(_file=results_file):
        truth.append(t)
        predictions.append(p)

    return __score(truth, predictions, extended=extended, label_set=label_set)


def print_model_performance_table(scores):
    print(f"Model \t& mac_p & mac_r & mac_f1 \t & mic_p & mic_r & mic_f1 \\\\")
    for stem, score in scores.items():
        if 'fine' in stem:
            continue
        print(f"{stem} \t& {score['all']['mac_p']} & {score['all']['mac_r']} & {score['all']['mac_f1']}"
              f"\t& {score['all']['mic_p']} & {score['all']['mic_r']} & {score['all']['mic_f1']} \\\\")

    print(f"Model \t& mac_p & mac_r & mac_f1 \t & mic_p & mic_r & mic_f1 \\\\")
    for stem, score in scores.items():
        if 'coarse' in stem:
            continue
        print(f"{stem} \t& {score['all']['mac_p']} & {score['all']['mac_r']} & {score['all']['mac_f1']}"
              f"\t & {score['all']['mic_p']} & {score['all']['mic_r']} & {score['all']['mic_f1']} \\\\")


def print_property_evaluation_table(scores):
    print("macro - micro")
    print(f"Model \t& all & 512 & 4096 &  16k & 16k_plus & support_2 & \topen  & closed & "
          f"\tall \t & 512 & 4096 & 16k & 16k_plus \t & support_2 \t & open & closed"
          f"\t& top_f1 & mid_f1 & bot_f1"
          f"\\\\")
    for stem, score in scores.items():
        print(f"{stem} \t& {score['all']['mac_f1']} & {score['512']['mac_f1']} & {score['4096']['mac_f1']} & "
              f"{score['16k']['mac_f1']} & {score['16k_plus']['mac_f1']} & "
              f"\t{score.get('open', {}).get('mac_f1', '--')} & {score.get('closed', {}).get('mac_f1', '--')} & "
              f"\t{score['support_2']['mac_f1']} & "
              f"\t{score['all']['mic_f1']}"
              f"\t & {score['512']['mic_f1']} & {score['4096']['mic_f1']}"
              f" &  {score['16k']['mic_f1']} & {score['16k_plus']['mic_f1']}"
              f"\t & {score.get('open', {}).get('mic_f1', '--')} "
              f"& {score.get('closed', {}).get('mic_f1', '--')} "
              f"\t & {score['support_2']['mic_f1']}"
              f"\t& {score['all'].get('sections', {}).get('top_f1', '--')} "
              f"& {score['all'].get('sections', {}).get('mid_f1', '--')} "
              f"& {score['all'].get('sections', {}).get('bot_f1', '--')} "
              f"\\\\")


def print_class_wise_scores_table(scores):
    class_wise_scores = {}
    for stem, score in scores.items():
        print(stem)
        for warning, _scores in score['all']['classes'].items():
            class_wise_scores.setdefault(warning, []).append(str(_scores['f1']))
    # print(class_wise_scores)
    for warning in warnings_ordered_by_frequency_coarse:
        print(f"{warning} & {'   & '.join(class_wise_scores[warning])} \\\\ ")
    print("\\midrule")
    for warning in warnings_ordered_by_frequency_fine:
        print(f"{warning} & {'   & '.join(class_wise_scores[warning])} \\\\ ")


@click.option('--test-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default="/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/predictions",
              help='Path to a directory with prediction (jsonl) files.')
@click.option('--metadata-file', type=click.Path(exists=True), help="Path to the trained model",
              default="/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/jsonl_datasets/v4-acl23/acl-final-070323-release.jsonl")
@click.option('--ff-map', type=click.Path(exists=True), help="",
              default="/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/tag-classification/v3/tag-classification-040523.csv")
@click.command()
def evaluate(test_dir: str, metadata_file: str, ff_map: str):
    """ """
    metadata = load_metadata(Path(metadata_file), tag_map=Path(ff_map))
    metadata = {work.work_id: work for work in metadata}
    scores = {}

    open_filter = partial(__filter_openness, return_open=True)
    closed_filter = partial(__filter_openness, return_open=False)
    len_filter_512 = partial(__filter, metadata=metadata, length=(0, 512))
    len_filter_4096 = partial(__filter, metadata=metadata, length=(512, 4096))
    len_filter_16k = partial(__filter, metadata=metadata, length=(4096, 16000))
    len_filter_16k_plus = partial(__filter, metadata=metadata, length=(16000, 93000))
    support_filter_2 = partial(__filter, metadata=metadata, min_support=2)
    support_filter_3 = partial(__filter, metadata=metadata, min_support=3)
    support_filter_5 = partial(__filter, metadata=metadata, min_support=5)

    for results_file in Path(test_dir).glob("*.jsonl"):
        logging.info(f"Processing results for {results_file.stem}")
        label_set = tw_coarse if 'coarse' in results_file.stem else tw_fine_open
        scores[results_file.stem] = {
            "all": _evaluate(results_file, label_set, extended=True),
            "512": _evaluate(results_file, label_set, len_filter_512),
            "4096": _evaluate(results_file, label_set, len_filter_4096),
            "16k": _evaluate(results_file, label_set, len_filter_16k),
            "16k_plus": _evaluate(results_file, label_set, len_filter_16k_plus),
            "support_2": _evaluate(results_file, label_set, support_filter_2),
            "support_3": _evaluate(results_file, label_set, support_filter_3),
            "support_5": _evaluate(results_file, label_set, support_filter_5),
        }
        if 'fine' in results_file.stem:
            scores[results_file.stem]["open"] = _evaluate(results_file, label_set, open_filter)
            scores[results_file.stem]["closed"] = _evaluate(results_file, label_set, closed_filter)

    open("evaluation-results.json", 'w').write(json.dumps(scores, indent=4))
    print_model_performance_table(scores)
    print_property_evaluation_table(scores)
    print_class_wise_scores_table(scores)


if __name__ == "__main__":
    evaluate()
