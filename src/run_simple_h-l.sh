#!/usr/bin/env bash
for lang in he hi hr hu id it kk la lv
do
python run_simple --dynet-seed 113 --lang $lang > trace_simple_baseline.$lang 2>&1
done