import logging
from typing import Tuple, Iterable, Dict, List
import click
from pathlib import Path
import json

from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from datetime import datetime as dt
from scipy.sparse import vstack
import numpy as np
from acl23_trigger_warning_assignment.util import _time, load_data

import joblib
import xgboost as xgb

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE These are the parameters for the ablation study. These are passed as parameters to the train_func
ABL_MODEL_PARAM = {"svm": {'c': [0.1, 0.2, 0.5, 1.0, 2.0]},
                   'xgboost': {'max_depth': [2, 3, 4],
                               'learning_rate': [0.25, 0.5, 0.75]}
                   }

ABL_VEC_PARAMS = [
    {'dir': 'word-1-all', 'n_gram_range': (1, 1), 'analyzer': 'word', 'f_select': 'None'},
    {'dir': 'word-3-all', 'n_gram_range': (3, 3), 'analyzer': 'word', 'f_select': 'None'},
    {'dir': 'word-13-chi2', 'n_gram_range': (1, 3), 'analyzer': 'word', 'f_select': 'chi2'},
    {'dir': 'word-15-chi2', 'n_gram_range': (1, 5), 'analyzer': 'word', 'f_select': 'chi2'}]


def dummy_preprocessor(x):
    return x


def fit_vectorizer(x_text, y, savepoint: Path, fit: bool = False,
                   n_gram_range=(3, 3), min_df=5, f_select='None', k=20000, **kwargs) -> Iterable:
    """
    This function builds the vectorizer (which transforms a the document into a BoW - vector).

    Check here for a general explanation: https://webis.de/lecturenotes-new.html#unit-en-text-representation

    The feature selection option can be used to remove some (unimportant) elements from the vector, so the remaining
      features have a bigger impact.


    :param x_text: the texts
    :param y: the labels
    :param savepoint: Path where to save the vectorizer
    :param fit: if true, we load the vectorizer from savepoint. If false, we fit a new one and save it.
    :param n_gram_range: vectorizer parameter, set here for ablation
    :param min_df: vectorizer parameter, set here for ablation
    :param f_select: feature selection to use. 'None" or 'chi2'
    :param k: Max feature limit for f_select
    :return:
    """
    if fit:
        logger.debug("fit vectorizer")
        vec = TfidfVectorizer(lowercase=False, ngram_range=n_gram_range, min_df=min_df,
                              token_pattern=None, tokenizer=dummy_preprocessor, preprocessor=dummy_preprocessor)
        x = vec.fit_transform(x_text)
        _time()
        joblib.dump(vec, savepoint / "vectorizer.joblib")

        if f_select is None or f_select == 'None':
            return x

        logger.debug("fit feature selection")
        if f_select == 'chi2':
            feature_selector = SelectKBest(chi2, k=k)  #
        else:
            raise AttributeError(f"f_select can not be {f_select}")
        x = feature_selector.fit_transform(x, y)
        _time()
        joblib.dump(feature_selector, savepoint / "feature-selector.joblib")
    else:
        logger.debug("load vectorizer")
        vec = joblib.load(savepoint / "vectorizer.joblib")
        logger.debug("load feature selection")
        x = vec.transform(x_text)
        if f_select == 'chi2':
            feature_selector = joblib.load(savepoint / "feature-selector.joblib")
            x = feature_selector.transform(x)

    return x


def _train_svm(x_train, y_train, x_validation, y_validation, savepoint: Path, ablate=True,
               c: List[int] = None, **kwargs) -> Tuple[callable, Dict]:
    """
    This function trains a XGB model on the given data.

    If ablate == True, it will run a hyperparameter search (currently a grid search over the parameters in ABL_MODEL_PARAM)

    :param x_train: training feature matrix (scipy sparse matrix)
    :param y_train: training labels (numpy array)
    :param x_validation: validation features (scipy sparse matrix)
    :param y_validation: validation labels (numpy array)
    :param savepoint: where to save the trained model to
    :param ablate: run the ablation study (gridsearch over multiple values)
    :return: (y_predicted, parameters) the predicted labels on the validation split
    """
    _time(True)
    parameters = {"c": 1}
    clf = OneVsRestClassifier(LinearSVC(C=1))
    if ablate:
        split_index = [-1] * len(y_train) + [0] * len(y_validation)
        x = vstack((x_train, x_validation))
        y = np.vstack((y_train, y_validation))
        ps = PredefinedSplit(test_fold=split_index)
        gs = GridSearchCV(clf, param_grid={'estimator__C': c}, verbose=1, cv=ps, n_jobs=5, scoring='f1_macro')
        gs.fit(x, y)
        logger.info(f"Best score in grid search: {gs.best_score_}")
        logger.info(f"Best parameters in grid search: {gs.best_params_}")
        be = gs.best_estimator_
        parameters = gs.best_params_
    else:
        clf.fit(x_train, y_train)
        be = clf

    _time()
    logger.info(f"save model to {savepoint}")
    joblib.dump(be, savepoint / "clf-ovr.joblib")
    return be, parameters


def _train_xgboost(x_train, y_train, x_validation, y_validation, savepoint: Path, ablate=True,
                   max_depth: List[int] = None, learning_rate: List[float] = 0.25, n_estimators: List[int] = None,
                   **kwargs) -> Tuple[callable, Dict]:
    """
    This function trains a XGB model on the given data.

    If ablate == True, it will run a hyperparameter search (currently a grid search over the parameters in ABL_MODEL_PARAM)

    :param x_train: training feature matrix (scipy sparse matrix)
    :param y_train: training labels (numpy array)
    :param x_validation: validation features (scipy sparse matrix)
    :param y_validation: validation labels (numpy array)
    :param savepoint: where to save the trained model to
    :param ablate: run the ablation study (gridsearch over multiple values)
    :return: (y_predicted, parameters) the predicted labels on the validation split
    """
    _time(True)
    parameters = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': learning_rate}
    clf = xgb.XGBClassifier(tree_method="hist", n_estimators=parameters['n_estimators'],
                            early_stopping_rounds=10,
                            max_depth=parameters['max_depth'], learning_rate=parameters['learning_rate'], n_jobs=16)

    if ablate:
        split_index = [-1] * len(y_train) + [0] * len(y_validation)
        x = vstack((x_train, x_validation))
        y = np.vstack((y_train, y_validation))
        ps = PredefinedSplit(test_fold=split_index)
        gs = GridSearchCV(clf, {'max_depth': max_depth,
                                'learning_rate': learning_rate}, verbose=1, cv=ps, scoring='f1_macro')
        gs.fit(x, y, eval_set=[(x_validation, y_validation)])
        logger.info(f"Best score in grid search: {gs.best_score_}")
        logger.info(f"Best parameters in grid search: {gs.best_params_}")
        be = gs.best_estimator_
        parameters = gs.best_params_
    else:
        clf.fit(x_train, y_train, eval_set=[(x_validation, y_validation)])
        be = clf

    _time()
    logger.info(f"save model to {savepoint}")
    joblib.dump(be, savepoint / "clf-ovr.joblib")
    return be, parameters


def run_trainer(training_dataset: Path, validation_dataset: Path, test_dataset: Path,
                savepoint: Path, model: str, ablate=False):
    """ Train the model. Here we also control the ablation.

    Ablated parameters are:
        - data sample: (fixed a-priory and passed via training_dataset_dir and validation_dataset_dir)
        - tokenizer:  n_gram_range, analyzer, f_select
        - Model: grid search over 'max_depth', 'learning_rate', and 'n_estimators'

    :param model: which model to train: `svm` or `xgboost`
    :param training_dataset: Path to a jsonl file with training data
    :param validation_dataset: Path to a jsonl file with validation data
    :param test_dataset: Path to a jsonl file with test data
    :param savepoint: Where to save the model, vectorizer, results to
    :param ablate: If True, run the ablation study.
    :return: None
    """
    logger.debug(f"Run Ablation: {ablate} for model {model}")

    model_params = ABL_MODEL_PARAM[model]
    if model == 'svm':
        train_func = _train_svm
    else:
        train_func = _train_xgboost

    def _run(xt, xv, xtt, vectorizer_params: Dict, model_params: Dict, savepoint: Path):
        savepoint = savepoint if not vectorizer_params["dir"] else savepoint / vectorizer_params["dir"]
        savepoint.mkdir(exist_ok=True)
        logger.debug("fit training vectorizer")
        x_train = fit_vectorizer(xt, y_train, savepoint, fit=True, **vectorizer_params)
        logger.debug("vectorize validation data")
        x_validation = fit_vectorizer(xv, y_validation, savepoint, **vectorizer_params)
        x_test = fit_vectorizer(xtt, y_test, savepoint, **vectorizer_params)

        logger.info("train model")
        best_model, parameters = train_func(x_train, y_train, x_validation, y_validation,
                                            savepoint, ablate=ablate, **model_params)

        y_validation_predicted = best_model.predict(x_validation)

        logger.info(f"Vectorizer Parameters: {vectorizer_params}")
        logger.info(f"Model Parameters: {parameters}")
        micro_f1 = f1_score(y_validation, y_validation_predicted, average='micro')
        macro_f1 = f1_score(y_validation, y_validation_predicted, average='macro')
        logger.info(f"trained with validation scores of {macro_f1} macro f1 and {micro_f1} micro f1")
        logger.info(f"Classification report on the validation data: {classification_report(y_validation, y_validation_predicted)}")
        results = {**vectorizer_params, **parameters, "micro_f1": micro_f1, "macro_f1": macro_f1}
        open(savepoint / 'validation-results.json', 'w').write(json.dumps(results))

        y_test_predicted = best_model.predict(x_test)

        open(savepoint / 'test-results.jsonl', 'w').writelines(
            [f"{json.dumps({'work_id': i, 'truth': _truth.tolist(), 'predictions': _predictions.tolist()})}\n"
             for i, _truth, _predictions in zip(ids_test, y_test, y_test_predicted)])

    logger.info("load training data")
    _, x_train_text, y_train = load_data(training_dataset)
    logger.info("load validation data")
    _, x_validation_text, y_validation = load_data(validation_dataset)
    logger.info("load test data")
    ids_test, x_test_text, y_test = load_data(test_dataset)

    if ablate:
        for vectorizer_parameter in ABL_VEC_PARAMS:
            _run(x_train_text, x_validation_text, x_test_text, vectorizer_parameter,
                 model_params, savepoint)

    else:
        _run(x_train_text, x_validation_text, x_test_text, {'dir': None}, {}, savepoint=savepoint)


@click.option('--training', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/train.jsonl",
              help='Path to the training dataset jsonl file')
@click.option('--validation', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/validation.jsonl",
              help='Path to the validation dataset jsonl file')
@click.option('--test', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../resources/dev-data/test.jsonl",
              help='Path to the test dataset jsonl file')
@click.option('-s', '--savepoint', type=click.Path(exists=False, file_okay=False),
              default="./models/xgb-baseline",
              help="Path where to store the trained model. Will be overwritten if it already exists.")
@click.option('--model', type=str, default='svm', help='`svm` or `xgboost``')
@click.option('-a', '--ablate', type=bool, default=False, is_flag=True,
              help='If set, run the ablation study.')
@click.command()
def run_experiment(training: str, validation: str, test: str, savepoint: str, model: str, ablate: bool):
    """
    Use the following command to train a model and save it to the default directoy.
    MODEL="xgboost"
    DATA="fine"
    SIZE="10k"
    SAVEPOINT="./xgb-10k-fine"

    poetry run python3 feature_engineering_experiments.py -a \
           --training "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/acl23-$DATA-open/train-$SIZE.jsonl" \
           --validation "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/acl23-$DATA-open/validation.jsonl" \
           --test /mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/multi-label-classification/dataset/acl23-final/data/acl23-$DATA-open/test.jsonl \
           --savepoint $SAVEPOINT \
           --model $MODEL
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_trainer(Path(training), Path(validation), Path(test), Path(savepoint), model, ablate)


if __name__ == "__main__":
    run_experiment()
