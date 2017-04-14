#### Example of using bilty from within code
##
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

import argparse


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

parser = argparse.ArgumentParser(description="""corpus stats""")
parser.add_argument("--lang", help="lang prefix", required=False)
args = parser.parse_args()


BASE="/projdata/alpage2/hmartine/data/ud1.3/orgtok/goldpos/"

train_data = BASE+args.lang + "-ud-train.conllu"
dev_data = BASE + args.lang + "-ud-dev.conllu"
test_data = BASE + args.lang + "-ud-test.conllu"

train_counter = getwords(train_data)
dev_counter = getwords(dev_data)
test_counter = getwords(test_data)
all_corpus = train_counter + test_counter + dev_data

def cover_instance(lexicon,wordcounter):
    filteredcounter = Counter(dict([(k,v) for k,v in wordcounter.items() if k in lexicon]))
    #print(sum(filteredcounter.values()), sum(wordcounter.values()), len(lexicon))
    #print(filteredcounter.most_common(15))
    #print(wordcounter.most_common(15))
    return sum(filteredcounter.values()) / sum(wordcounter.values())

def cover_type(lexicon,wordcounter):
    return len(lexicon.intersection(set(wordcounter.keys()))) / len(set(wordcounter.keys()))

def TTR(wordcounter):
    return len(wordcounter.keys) / sum(wordcounter.values())

def getmetrics(lang,train_counter,dev_counter,test_counter,all_corpus):
    metrics = dict()
    metrics["OOTC_instance"] = cover_instance(set(train_counter.keys()),test_counter)
    metrics["OOTC_type"] = cover_instance(set(train_counter.keys()),test_counter)
    metrics["TTR"] = TTR(all_corpus)
    metrics["size_dev"] = sum(dev_counter.values())
    metrics["size_train"] = sum(test_counter.values())
    metrics["size_test"] = sum(train_counter.values())
    print("\t".join(sorted(metrics.keys())))
    print("\t".join([str(metrics[k]) for k in sorted(metrics.keys())]))

getmetrics(args.lang,train_counter,dev_counter,test_counter,all_corpus)
