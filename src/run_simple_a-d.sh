#!/usr/bin/env bash
for lang in ar bg ca cs cu da de
do
python run_simple.py  --lang $lang > trace_simple_baseline.$lang 2>&1
done