# Legacy demo (not part of the main pipeline)

`etl.py` builds a small **synthetic** star schema (100 made-up customers, one year of random 2025 transactions)
and writes the CSV files in this folder. It was an early prototype of a data-pipeline stage.

The real pipeline (`train.py` and `retention_engine.py`) reads the Online Retail II transactions directly and
does not use anything in this folder. The numbers in these CSVs are random and say nothing about the model.
