#!/usr/bin/env bash
for lang in ar bg ca cs cu da de el en es et eu fa fi fr ga gl got grc he hi hr hu id it kk la lv nl no pl pt ro ru sl sv ta tr zh
do
python run_simple_embeds.py --lang $lang > trace_simple_embeds.$lang 2>&1
done