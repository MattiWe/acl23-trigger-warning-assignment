import sys
import requests
from dataclasses import asdict
from datetime import datetime, date
import json

from parser import build_work, Ao3PageError
import click
import tqdm
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


@click.group()
def cli():
    pass


@click.option('-p', '--dataset_path', type=click.Path(exists=True), default='../data',
              help="Path to the (dehydrated) dataset")
@cli.command()
def hydrate(dataset_path):
    work_ids = []
    for labels_file in Path(dataset_path).rglob("*-labels.jsonl"):
        work_ids.extend([json.loads(line)['work_id'] for line in open(labels_file)])

    with open(dataset_path / "works.jsonl", 'w') as of:
        for work_id in work_ids:
            work = {"work_id": work_id, "text": parse_work(work_id)}
            of.write(f"{json.dumps(work)}\n")


if __name__ == "__main__":
    cli()
