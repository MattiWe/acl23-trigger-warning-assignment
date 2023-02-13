import logging
import requests
from datetime import datetime, date
import json

from parser import build_work, Ao3PageError
import click
from tqdm import tqdm
from pathlib import Path


class WorkJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime) or isinstance(obj, date):
            return obj.isoformat()


def url_from_work_id(work_id):
    return f"https://archiveofourown.org/works/{work_id}?show_comments=true&view_adult=true&view_full_work=true"


def parse_work(work_id):
    try:
        r = requests.get(url_from_work_id(work_id))
        w = build_work(r.url, r.text)
        return w
    except Ao3PageError as e:
        print(f"Couldn't find a work with ID '{work_id}'. Please check the ID")
    except requests.exceptions.ConnectionError as e:
        print(f"Couldn't load a work with ID '{work_id}'. Please try again.")


@click.option('-p', '--dataset_path', type=click.Path(exists=True), default='../data',
              help="Path to the (dehydrated) dataset")
@click.command()
def hydrate(dataset_path: str):
    """ Download all works from AO3
        The label files contain different views of the same works, so many works overlap.
        To only download each work once (and to not keep them in-memory) we download them like:
            1. Load all IDs, deduplicate, and store which ID belongs to which split and which file.
            2. Download each work once and write them to each corresponding file.

        @param dataset_path: Directory labels files. Must be a 2-level hierarch, with *.jsonl files at the leaves, like:
        - <dataset_path>
          |- closed-set-coarse-grained
            |- *-labels.jsonl files
            |- ...
          | - opens-set-coarse-grained
            |- ...
          |- ...
     """
    dataset_path = Path(dataset_path)

    work_ids = set()
    file_names_by_work_id = {}
    work_files = []

    # 1. Find out all IDs
    for labels_file in tqdm(dataset_path.rglob("*-labels.jsonl"), desc='loading work IDs from the label files'):
        work_file_name = f"{labels_file.parent.stem}/{labels_file.stem.rstrip('-labels')}-fics.jsonl"
        work_files.append(work_file_name)
        wids = {json.loads(line)['work_id'] for line in open(labels_file)}
        for wid in wids:
            file_names_by_work_id.setdefault(wid, set()).add(work_file_name)
        work_ids.update(wids)

    output_file_by_file_name = {work_file_name: open(dataset_path / work_file_name, 'w')
                                for work_file_name in work_files}
    for work_id in tqdm(work_ids, desc='downloading works from AO3'):
        try:
            text = [c.content for c in parse_work(work_id).chapters]
        except (Ao3PageError, AttributeError):
            # If the works was deleted or set private, we just skip this entry and print a warning
            logging.warning(f"The work with ID {work_id} is not available for download anymore")
        else:
            work = {"work_id": work_id, "text": text}
            for of_filename in file_names_by_work_id[work_id]:
                output_file_by_file_name[of_filename].write(f"{json.dumps(work)}\n")

    for k, v in output_file_by_file_name.items():
        v.close()


if __name__ == "__main__":
    hydrate()
