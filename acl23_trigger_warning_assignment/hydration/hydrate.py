import logging
from typing import Union
import requests
from datetime import datetime, date
import traceback
import json

from acl23_trigger_warning_assignment.hydration.parser import build_work, Ao3PageError
import click
from tqdm import tqdm
from pathlib import Path
from wasabi import msg


class WorkJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime) or isinstance(obj, date):
            return obj.isoformat()


def url_from_work_id(work_id):
    return f"https://archiveofourown.org/works/{work_id}?show_comments=true&view_adult=true&view_full_work=true"


def parse_work(work_id):
    r = requests.get(url_from_work_id(work_id))
    w = build_work(r.url, r.text)
    return w


@click.option('-d', '--dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the (dehydrated) dataset")
@click.option('--splits', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to a directory with id-lists (to indicate individual splits")
@click.command()
def hydrate(dataset: str, splits: str):
    """ Hydrate the dataset. The hydrated data will be stored right next to the dehydrated version.

    poetry run python3 main.py hydrate \
        --dataset /Users/matti/Documents/data/trigger-warning-corpus/webis-trigger-warning-corpus-dehydrated.jsonl
    """
    run_hydrate(Path(dataset), Path(splits))


def run_hydrate(dataset: Path, split_dir: Union[Path, None]):
    """ Download all works from AO3 given in the dataset file and create the splits

        :param split_dir: Directory with .txt-files, each is a list of work_ids
        :param dataset: path to the .jsonl file of the dehydrated dataset
     """

    # 1. load split IDs and create file handles
    file_names_by_work_id = {}
    output_file_handles = {}
    with msg.loading('Loading splits ... '):
        for split_file in split_dir.glob("*"):
            split_name = split_file.stem
            output_file_handles[split_name] = open(dataset.parent / f"{split_name}.jsonl", 'w')
            for work_id in [line.strip() for line in open(split_file)]:
                file_names_by_work_id.setdefault(work_id, []).append(split_name)

    # 2. iterate the dataset, download the work, and write the result to the appropriate split
    with open(dataset.parent / "webis-trigger-warning-corpus.jsonl", 'w') as of_dehydrated:
        with msg.loading("Downloading works ... "):
            for work in open(dataset):
                work = json.loads(work)
                work_id = work["work_id"]

                try:
                    parsed_work = parse_work(work_id)
                    work["text"] = [c.content for c in parsed_work.chapters]
                    work["notes"] = parsed_work.notes.content if parsed_work.notes else ""
                    work["endnotes"] = parsed_work.endnotes.content if parsed_work.endnotes else ""
                    work["summary"] = parsed_work.summary.content if parsed_work.summary else ""
                except (Ao3PageError, AttributeError) as e:
                    # If the works was deleted or set private, we just skip this entry and print a warning
                    msg.warn(f"The work with ID {work_id} is not available for public download anymore")

                except requests.exceptions.ConnectionError as e:
                    msg.fail(f"Couldn't load a work with ID '{work_id}' due to a connection error: {e}. Please try again.")
                    print(traceback.format_exc())
                else:
                    of_dehydrated.write(f"{json.dumps(work)}\n")
                    for of_filename in file_names_by_work_id[work_id]:
                        output_file_handles[of_filename].write(f"{json.dumps(work)}\n")

    for k, v in output_file_handles.items():
        v.close()

    msg.text(f"Finished download. Skipped {msg.counts['warn']} works")


if __name__ == "__main__":
    hydrate()
