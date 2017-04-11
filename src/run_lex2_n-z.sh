#!/usr/bin/env bash
for lang in nl no pl pt ro ru sl sv ta tr zh
do
python run_simple_lex2.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_simple_lex2.$lang 2>&1
done