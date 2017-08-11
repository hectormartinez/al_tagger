import argparse

from collections import Counter


def readlex(lex):
    L = set()
    for line in open(lex,encoding="utf-8").readlines():
        line = line.strip().split("\t")
        L.add(line[0])
    return L

def process(infile,ref,lex,trainlex):
    totalwords = 0
    totaloovTbutL = 0
    totaloovT = 0
    correctoovT = 0
    totaloovL = 0

    correctoovL = 0
    error_counter = Counter()
    G = []
    P = []
    for line,original_line in zip(open(infile,encoding="utf-8").readlines(),open(ref,encoding="utf-8").readlines()):
        line = line.strip()
        if line:
            totalwords+=1
            w,g,p = line.split("\t")
            original_w, original_ = original_line.strip().split("\t")
            G.append(g)
            P.append(p)
            if g != p:
                error_counter[line]+=1

            if original_w not in trainlex:
                totaloovT+=1
                correctoovT+= int(g==p)
                if original_w in lex:
                    totaloovTbutL+=1
            if original_w not in lex:
                totaloovL += 1
                correctoovL += int(g == p)

    accOOTC= correctoovT/totaloovT #print(infile, "accuracy_for_OOTC_words",
    accOOTCinlex = correctoovL/totaloovL #print(infile, "accuracy_for_OOTC_in_lex",correctoovL/totaloovL)
    not_in_train = totaloovT/totalwords# print(infile, "non_in_train",totaloovT/totalwords)
    not_in_train_but_present_in_lex = totaloovTbutL/totalwords# print(infile, "non_in_train_but_present_in_lexicon",totaloovTbutL/totalwords)
    cov_not_in_lex =  totaloovL / totalwords #print(infile, "not_in_lexicon", totaloovL / totalwords)

    outlist = [accOOTC,accOOTCinlex,not_in_train,not_in_train_but_present_in_lex,cov_not_in_lex]
    outlist = [str(x) for x in outlist]
    return outlist


def main():
    parser = argparse.ArgumentParser(
        description="""Obtains additional scores for a certain tagged output. Mainly OOV""")
    parser.add_argument("--lang")
    parser.add_argument("--baseline", help="")
    parser.add_argument("--best", help="")
    parser.add_argument("--ref", help="")
    parser.add_argument("--lex", help="")
    parser.add_argument("--train_data", help="")

    args = parser.parse_args()
    lex = readlex(args.lex)
    trainlex = readlex(args.train_data)
    accOOTCbase, accOOTCinlexbase, not_in_train, not_in_train_but_present_in_lex, cov_not_in_lex=process(args.baseline,args.ref,lex,trainlex)
    figures=process(args.best,args.ref,lex,trainlex)

    print("\t".join([args.lang,accOOTCbase,accOOTCinlexbase]+figures))



if __name__ == "__main__":
    main()