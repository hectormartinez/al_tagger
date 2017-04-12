#!/usr/bin/env bash
for lang in bg da de cs
do
python run_lex2_embeds.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_simple_lex2.$lang 2>&1
done
