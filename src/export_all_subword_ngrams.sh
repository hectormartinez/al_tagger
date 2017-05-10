#!/usr/bin/env bash

for file in `ls /projdata/alpage2/hmartine/data/ud1.3/orgtok/goldpos/*`
do
for k in 2 3 4 5
do
    python create_baseline_subwords.py --infile $file --ngram $k
done
done