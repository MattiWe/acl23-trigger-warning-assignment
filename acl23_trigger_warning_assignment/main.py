import click
from pathlib import Path
from acl23_trigger_warning_assignment.classification.longformer_experiments import run_experiment
from acl23_trigger_warning_assignment.classification.feature_engineering_experiments import run_trainer
from acl23_trigger_warning_assignment.hydration.hydrate import run_hydrate
from acl23_trigger_warning_assignment.util import to_datasets


@click.group()
def cli():
    pass


@click.option('-c', '--checkpoint', type=click.STRING, default='allenai/longformer-base-4096',
              help="base checkpoint for model and tokenized")
@click.option('--training', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the training dataset jsonl file')
@click.option('--validation', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the validation dataset jsonl file')
@click.option('--test', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the test dataset jsonl file')
@click.option('-s', '--savepoint', type=click.Path(),
              default="./models/neural-baseline", help="Path to save the model in.")
@click.option('--epochs', type=click.INT, default=5, help="Path to save the model in.")
@click.option('--batches', type=click.INT, default=4, help="Batch size")
@click.option('--lr', type=click.FLOAT, default=0.00002, help="Initial learning rate")
@click.option('-n', '--name', type=click.STRING, default='develop', help="base name of the model (for wandb)")
@cli.command()
def neural_experiments(checkpoint: str, training: str, validation: str, test: str, savepoint: str,
                       epochs: int, batches: int, lr: float, name: str):
    """
    Run the (neural baseline) experiments.

    poetry run python3 main.py neural-experiments \
           --checkpoint "pre-trained huggingface checkpoint" \
           --training  <path/to/train.jsonl> \
           --validation <path/to/validation.jsonl> \
           --test <path/to/test.jsonl> \
           --savepoint <path/to/save/model/in> \
           --lr 2e-5 \
           -- batches 32 \
           -- epochs 3 \
           --name "name used for wandb"
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_experiment(checkpoint, training, validation, test, savepoint, epochs, batches, lr, name)


@click.option('--training', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the training dataset jsonl file')
@click.option('--validation', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the validation dataset jsonl file')
@click.option('--test', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to the test dataset jsonl file')
@click.option('-s', '--savepoint', type=click.Path(exists=False, file_okay=False),
              default="./models/feature-baseline",
              help="Path where to store the trained model. Will be overwritten if it already exists.")
@click.option('--model', type=str, default='svm', help='`svm` or `xgboost``')
@click.option('-a', '--ablate', type=bool, default=False, is_flag=True,
              help='If set, run the ablation study.')
@cli.command()
def feature_experiments(training: str, validation: str, test: str, savepoint: str, model: str, ablate: bool):
    """
    Run the feature baselines ()

    Use the following command to train a model and save it to the default directory.

    poetry run python3 main.py feature-experiments -a \
           --training  <path/to/train.jsonl> \
           --validation <path/to/validation.jsonl> \
           --test <path/to/test.jsonl> \
           --savepoint <path/to/save/model/in> \
           --model "`svm` or `xgboost"
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_trainer(Path(training), Path(validation), Path(test), Path(savepoint), model, ablate)


@click.option('-d', '--dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the (dehydrated) dataset")
@click.option('--splits', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default=None,
              help="Path to a directory with id-lists (to indicate individual splits). Works with these ids"
                   "will be stored in individual files (ready for `prepare_dataset`)")
@cli.command()
def hydrate(dataset: str, splits: str):
    """ Hydrate the dataset. The hydrated data will be stored right next to the dehydrated version.

    poetry run python3 main.py hydrate \
        --dataset /Users/matti/Documents/data/trigger-warning-corpus/webis-trigger-warning-corpus-dehydrated.jsonl
    """
    run_hydrate(Path(dataset), Path(splits))


@click.option('-i', '--input-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the (hydrated) dataset to prepare.")
@click.option('-o', '--output-file', type=click.Path(exists=False, file_okay=True, dir_okay=False),
              help="Path where to write the result file.")
@click.option('--label-set', type=str, default="warnings_fine_open",
              help="warnings_fine_open` or `warnings_coarse_open` --- whatever label-set should be encoded as `labels`.")
@cli.command()
def prepare_dataset(input_file: str, output_file: str, label_set: str):
    """ Convert the input_file from the hydrated dataset format to a huggingface (datasets) compatible variant
        with array-style labels and without metadata.

    poetry run python3 main.py prepare-dataset \
        --dataset /Users/matti/Documents/data/trigger-warning-corpus/webis-trigger-warning-corpus-dehydrated.jsonl
    """
    to_datasets(Path(input_file), Path(output_file), label_set)


if __name__ == "__main__":
    cli()
