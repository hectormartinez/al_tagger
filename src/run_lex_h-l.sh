#!/usr/bin/env bash
for lang in he hi hu id it kk lv
do
python run_simple_lex.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_simple_lex.$lang 2>&1
done