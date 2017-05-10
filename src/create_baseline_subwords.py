import argparse
import math


def segment_char_grams(word,k):
    segments = []
    word = "^"+word+"$"
    steps = math.ceil(len(word)/k)

    for i in range(1,steps+1):
        tailsize = -1*k*(i)
        segments.append(word[tailsize:][:k]) #retrieve the tail, and then the first 2 from the tail
    return segments


def main():
    parser = argparse.ArgumentParser(description="""Creates baseline morpho splits for words in corpus""")
    parser.add_argument("--infile", help="",default="../corpus/pos_ud_en_dev.2col")
    parser.add_argument("--ngram", help="",default=3,type=int)

    args = parser.parse_args()
    for line in open(args.infile).readlines()[:30]:
        line = line.strip()
        if line:
            word,pos = line.split("\t")
            print(" ".join(segment_char_grams(word,args.ngram)))

        else:
            print("")

if __name__=="__main__":
    main()
