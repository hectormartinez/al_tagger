#!/usr/bin/env bash
for lang in he hi hr hu id it kk la lv
do
python run_simple.py --lang $lang > trace_simple_baseline.$lang 2>&1
done