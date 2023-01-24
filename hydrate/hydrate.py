import click
import sys
import requests
from dataclasses import asdict
from datetime import datetime, date
import json

from parser import build_work, Ao3PageError

@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli()