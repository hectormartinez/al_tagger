#!/usr/bin/env bash
for lang in ar bg ca cs cu da de el en es et eu fa fi fr ga gl got grc he hi hr hu id it kk la lv nl no pl pt ro ru sl sv ta tr zh
do
python run_simple_nochars.py --dynet-seed 113 --lang $lang > trace_simple_nochars.$lang 2>&1
done