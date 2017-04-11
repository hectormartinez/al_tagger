import codecs
import numpy as np
import sys
from collections import defaultdict

def load_embeddings_file(file_name, sep=" ",lower=False):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name,encoding="utf-8"):
        fields = line.split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        emb[word] = vec

    print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
    return emb, len(emb[word])

def read_lexicon_file(infile,w2i,coarse_lex=0):
    L = dict()
    ft = defaultdict(set)
    tag_set = set()
    for line in open(infile,encoding="utf-8").readlines():
        line = line.strip()
        if line:
            try:
                form, tag, lemma = line.split("\t") #requires f,t,l format
            except:
                print("lexicon error:",line.split("\t"))
            if coarse_lex:
                if "#" in tag: tag = tag[:tag.index("#")]
            if form not in w2i:
                w2i[form] = len(w2i.keys())
            tag_set.add(tag)
            ft[form].add(tag)

    tag_index = sorted(tag_set)
    for form, possible_tags in ft.items():
        L[form] =  [1 if tag_index[i] in possible_tags else 0 for i in range(len(tag_index))]
    L["_UNK"] = list(np.ones(len(tag_index)))
    return L, len(tag_index), w2i

def read_conll_file(file_name):
    """
    read in conll file
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)

    """
    current_words = []
    current_tags = []
    
    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            if len(line.split("\t")) != 2:
                if len(line.split("\t")) == 1: # emtpy words in gimpel
                    raise IOError("Issue with input file - doesn't have a tag or token?")
                    #word = "|"
                    #tag = line.split("\t")[0]
                    #print(tag,file=sys.stderr)
                else:
                    print("erroneous line: {} (line number: {}) ".format(line, i), file=sys.stderr)
                    exit()
            else:
                word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            if current_words: #skip emtpy lines
                yield (current_words, current_tags)
            current_words = []
            current_tags = []

        
    # check for last one
    if current_tags != []:
        yield (current_words, current_tags)
            
    
if __name__=="__main__":
    allsents=[]
    unique_tokens=set()
    unique_tokens_lower=set()
    for words, tags in read_conll_file("data/gimpel.train"):
        allsents.append(words)
        unique_tokens.update(words)
        unique_tokens_lower.update([w.lower() for w in words])
    assert(len(allsents)==1002)
    assert(len(unique_tokens)==4396)
    assert(len(unique_tokens_lower)==3742)
    #emb,l = load_embeddings_file("data/head_senna") #first 10 senna.txt emb
    #assert(l==50)
    #print(emb["#"])
