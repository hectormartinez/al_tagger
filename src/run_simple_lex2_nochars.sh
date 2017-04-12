#!/usr/bin/env bash
for lang in bg da de el en es et fa fr ga gl grc hr hu id it la nl no pl pt ro sl sv cs
do
python run_simple_lex2_nochars.py --dynet-seed 113 --lang $lang > trace_lex2_nochars.$lang 2>&1
done
