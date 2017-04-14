#### Example of using bilty from within code
##
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

import argparse
import random
### Use --dynet-seed $SEED

import os.path

from collections import Counter,defaultdict

def getwords(infile):
    C = Counter()
    for line in open(infile,encoding="utf-8"):
        line = line.strip()
        if line:
            a = line.split("\t")
            C[a[0]]+=1
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
dev_counter = getwords(dev_data)
test_counter = getwords(test_data)
traintest_counter = train_counter + test_counter

def cover_instance(lexicon,wordcounter):
    filteredcounter = Counter(dict([(k,v) for k,v in wordcounter.items() if k in lexicon]))
    #print(sum(filteredcounter.values()), sum(wordcounter.values()), len(lexicon))
    #print(filteredcounter.most_common(15))
    #print(wordcounter.most_common(15))
    return sum(filteredcounter.values()) / sum(wordcounter.values())

def cover_type(lexicon,wordcounter):
    return len(lexicon.intersection(set(train_counter.keys()))) / len(set(wordcounter.keys()))

def getmetrics(lang, lexicon, lexfile,train_counter,dev_counter,test_counter,traintest_counter):
    metrics = dict()
    metrics["lang\tlex"] = lang + "\t" + lexicon
    metrics["cover_type_train"] = "_"
    metrics["cover_type_dev"] = "_"
    metrics["cover_type_test"] = "_"
    metrics["cover_type_traintest"] = "_"
    metrics["cover_instance_train"] = "_"
    metrics["cover_instance_dev"] = "_"
    metrics["cover_instance_test"] = "_"
    metrics["cover_instance_traintest"] = "_"
    
    if os.path.isfile(lexfile):
            lex = set(getwords(lexfile).keys())
            metrics["cover_type_train"] = cover_type(lex, train_counter)
            metrics["cover_type_dev"] = cover_type(lex, dev_counter)
            metrics["cover_type_test"] = cover_type(lex, test_counter)
            metrics["cover_type_traintest"] = cover_type(lex, traintest_counter)
            metrics["cover_instance_train"] = cover_instance(lex, train_counter)
            metrics["cover_instance_dev"] = cover_instance(lex, dev_counter)
            metrics["cover_instance_test"] = cover_instance(lex, test_counter)
            metrics["cover_instance_traintest"] = cover_instance(lex, traintest_counter)
            print("\t".join(sorted(metrics.keys())))
            print([str(metrics[k]) for k in sorted(metrics.keys())])

getmetrics(args.lang, "lex", lexfile,train_counter,dev_counter,test_counter,traintest_counter)
getmetrics(args.lang, "lex2", lexfile2,train_counter,dev_counter,test_counter,traintest_counter)
