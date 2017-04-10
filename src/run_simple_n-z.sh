#!/usr/bin/env bash
for lang in nl no pl pt ro ru sl sv ta tr zh
do
python run_simple.py  --lang $lang > trace_simple_baseline.$lang 2>&1
done