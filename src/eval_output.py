import argparse
parser = argparse.ArgumentParser(description="""Obtains additional scores for a certain tagged output. Mainly OOV""")
parser.add_argument("--infile", help="lang prefix")
args = parser.parse_args()

#Obtain OOV scores
totalwords = 0
totaloov = 0
correctoov = 0
for line in open(args.infile,encoding="utf-8"):
    line = line.strip()
    if line:
        totalwords+=1
        w,g,p = line.split("\t")
        if w == "_UNK": #it's OOV as determined by the system
            totaloov+=1
            correctoov+= int(g==p)

print(totaloov/totalwords, correctoov/totaloov)