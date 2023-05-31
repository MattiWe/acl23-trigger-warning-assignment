# acl23-trigger-warning-assignment

This repository contains the data and code for **Trigger Warning Assignment as a Multi-Label Document Classification Problem**

## Get the Data

1. Download the dehydrated data from zenodo: https://doi.org/10.5281/zenodo.7976807.

2. Hydrate the data:

   ```
   poetry install
   poetry run python3 acl23_trigger_warning_assignment/main.py hydrate \
     -d <path/to/dehydrated/dataset> \
     --splits "acl23_trigger_warning_assignment/resources/splits"
   ```
   
The hydrator will scrape the work text from AO3 and complete the dataset. 
NOTE: If a work is not available anymore, the document will be skipped and the ID is logged. If you need a complete version of the data, please contact us directly.

This download might take a long time.

`/resources/splits` contains lists of work_ids, indicating which work from the dataset was included in which data split.


## Labels

`/resources` contains the developed label sets and the grouped institutional source data for reproducibility. 
`/resources/institutional-warnings` contains the sources of the labels for reproduction or modification 


## Prepare the data for experimentation

Call the following script on each of the splits from the step above to preprocess the texts and convert the dataset into the format used by huggingface. 

   ```
   poetry run python3 acl23_trigger_warning_assignment/main.py prepare-dataset \ 
     --input-file <path/to/any.jsonl> \ 
     --output-file <path/to/any/other.jsonl> \ 
     --label-set 'warnings_coarse_open'
   ```

See the `--help` for further explanation. 


## Experiments

Re-run the experimental evaluation:

   ```
   poetry run python3 acl23_trigger_warning_assignment/main.py feature-experiments -a \
     --training  <path/to/train.jsonl> \
     --validation <path/to/validation.jsonl> \
     --test <path/to/test.jsonl> \
     --savepoint <path/to/save/model/in> \
     --model "`svm` or `xgboost"
   ```

or 

   ```
   poetry run python3 acl23_trigger_warning_assignment/main.py neural-experiments \
     --checkpoint "pre-trained huggingface checkpoint" \
     --training  <path/to/train.jsonl> \
     --validation <path/to/validation.jsonl> \
     --test <path/to/test.jsonl> \
     --savepoint <path/to/save/model/in> \
     --lr 2e-5 \
     -- batches 32 \
     -- epochs 3 \
     --name "name used for wandb"
   ```
