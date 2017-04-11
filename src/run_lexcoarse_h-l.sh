#!/usr/bin/env bash
for lang in he hi hr hu id it kk la lv
do
python run_simple_lex_coarse.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_simple_lexcoarse.$lang 2>&1
done