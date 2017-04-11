#!/usr/bin/env bash
for lang in ar bg ca cs da de el en es et eu fa fi fr ga he hi hr id it nl no pl pt sl sv
do
python run_simple_embeds.py --dynet-seed 113 --lang $lang > trace_simple_embeds.$lang 2>&1
done



