#!/usr/bin/env bash
for lang in hr id it
do
python run_lex2_embeds.py --dynet-mem 2000 --dynet-seed 113 --lang $lang > trace_lex2_embeds.$lang 2>&1
done
