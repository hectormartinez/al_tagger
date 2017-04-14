#### Example of using bilty from within code
##
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

from simplebilty_nochars import SimpleBiltyTaggerNoChars
import argparse
import random
### Use --dynet-seed $SEED

import os.path
os.path.isfile("/Users/hector/data/basquebible/Ab_99")

from collections import Counter

def getwords(infile):
    C = Counter
    for line in open(infile,encoding="utf-8"):
        line = line.strip()
        if line:
            tokenlist = line.split("\n")
            C[tokenlist[0]]+=1
    return C

parser = argparse.ArgumentParser(description="""Run the NN tagger""")
parser.add_argument("--lang", help="lang prefix", required=False)
args = parser.parse_args()


BASE="/projdata/alpage2/hmartine/data/ud1.3/orgtok/goldpos/"
LEXBASE="/projdata/alpage2/hmartine/data/lex/"

train_data = BASE+args.lang + "-ud-train.conllu"
dev_data = BASE + args.lang + "-ud-dev.conllu"
test_data = BASE + args.lang + "-ud-test.conllu"
lexfile = LEXBASE + args.lang + ".lex"
lexfile2 = LEXBASE + args.lang + ".lex2"

train_counter = getwords(train_data)
dev_counter = getwords(train_data)
test_counter = getwords(train_data)
traintest_counter = train_counter + test_counter

if os.path.isfile(lexfile):
    lex = getwords(lexfile)
    metrics = []
    metrics.append(set(lex.keys).intersection(set(train_counter.keys)))
    