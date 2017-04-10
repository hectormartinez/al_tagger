#!/usr/bin/env bash
for lang in el en es et eu fa fi fr ga gl got grc
do
python run_simple --dynet-seed 113 --lang $lang > trace_simple_baseline.$lang 2>&1
done