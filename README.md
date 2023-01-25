# acl23-trigger-warning-assignment

This repository contains the data and code for the ACL submission (under review) for **Trigger Warning Assignment as a Multi-Label Document Classification Problem**

## Data

`data\` contains the 4 datasets used in the study. This repo contains only the labels. The work text should be scraped from AO3 for ethical reasons. Use the hydration utilities to download them. 

These files will be moved to zenodo after the review phase. 


## Hydrate

`hydrate/` contains a script to scrape the work text from AO3 and complete the dataset. If a work is not available anymore, the text will be empty.

    $ pip install -r requirements.txt
    $ python3 hydrate.py hydrate

## Experiments

`experiments/` contains the scripts we used to run the experiments described in the paper.
`experiments/utils.py` contains code to pre-process the datasets into the format used by the model scripts. 

