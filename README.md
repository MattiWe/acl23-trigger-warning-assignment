# acl23-trigger-warning-assignment

This repository contains the data and code for the ACL submission (under review) for **Trigger Warning Assignment as a Multi-Label Document Classification Problem**

## Data

## Hydrate

`hydrate/` contains a script to scrape the work text from AO3 and complete the dataset. If a work is not available anymore, the text will be empty.

    $ pip install -r requirements.txt
    $ python3 hydrate.py hydrate

## Experiments

`experiments/` contains the scripts we used to run the experiments described in the paper. 

