#!/usr/bin/env bash
for lang in ar bg ca cu da de cs
do
python run_simple_lex.py --dynet-seed 113 --lang $lang > trace_simple_lex.$lang 2>&1
done