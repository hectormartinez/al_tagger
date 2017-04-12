#!/usr/bin/env bash
for lang in nl no pl pt sl sv
do
python run_lex2_embeds.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_simple_lex2.$lang 2>&1
done
