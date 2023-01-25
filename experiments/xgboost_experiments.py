import json
import logging
from tqdm import tqdm
import click
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from joblib import dump, load
import xgboost as xgb
from util import _evaluate


def _load_train_data(dataset_dir: Path, savepoint: Path):
    def _load(filename):
        text = []
        label = []
        for line in tqdm(open(filename), desc=f"loading examples from {filename}"):
            line = json.loads(line)
            text.append(" ".join([str(x) for x in line["text"]]))
            label.append(line["labels"])
        return text, label

    def tokenize(x: str):
        return x.split(" ")

    training_text, y_train = _load(dataset_dir / "training.jsonl")
    validation_text, y_validation = _load(dataset_dir / "validation.jsonl")
    test_text, y_test = _load(dataset_dir / "test.jsonl")

    vec = TfidfVectorizer(lowercase=False, tokenizer=tokenize, analyzer='word',
                          ngram_range=(1, 3), min_df=3)
    feature_selector = SelectKBest(chi2, k=10000)
    x_train = vec.fit_transform(training_text)
    x_train = feature_selector.fit_transform(x_train, y_train)
    x_validation = vec.transform(validation_text)
    x_validation = feature_selector.transform(x_validation)
    x_test = vec.transform(test_text)
    x_test = feature_selector.transform(x_test)
    dump(vec.vocabulary_, savepoint / "vectorizer_vocab.joblib")

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def _train_model(x_train, y_train, x_validation, y_validation, savepoint: Path):
    logging.warning("train model")
    # Given a sample with 3 output classes and 2 labels, the corresponding y should be encoded as [1, 0, 1]

    clf = xgb.XGBClassifier(tree_method="hist", n_estimators=50, max_depth=2, eta=1, n_jobs=8)

    # gs = GridSearchCV(clf, {'max_depth': [2, 4, 6],
    #                         'n_estimators': [50, 100, 200]}, verbose=1,
    #                         n_jobs=4)
    clf.fit(x_train, y_train, eval_set=[(x_validation, y_validation)])
    # save model
    dump(clf, savepoint / "clf-ovr.joblib")

    # return gs.best_estimator_
    return clf


def run_experiment(dataset_dir: str, savepoint: str):
    logging.warning("load training data")
    x_train, y_train, x_validation, y_validation, x_test, y_test = _load_train_data(Path(dataset_dir), Path(savepoint))
    model = _train_model(x_train, y_train, x_validation, y_validation, Path(savepoint))

    y_test_predictions = model.predict(x_test)
    logging.warning("evaluate test")
    scores = _evaluate(y_test, y_test_predictions)
    open(Path(savepoint) / "test-scores.json", 'w').write(json.dumps(scores))
    logging.warning("test scores", scores)


@click.group()
def cli():
    pass


@click.option('-d', '--dataset-dir', type=click.Path(exists=True),
              help="Path where the training.jsonl and validation.jsonl is.")
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the model in.")
@cli.command()
def train(dataset_dir, savepoint):
    """
    $ python3 xgboost_experiments.py train \
        -d <input-dataset> \
        -s "./models"
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_experiment(dataset_dir, savepoint)
    logging.info(f"trained model at {savepoint}")


if __name__ == "__main__":
    cli()
