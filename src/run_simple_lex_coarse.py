#### Example of using bilty from within code
## 
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

from simplebilty import SimpleBiltyTagger
import argparse
import random
### Use --dynet-seed $SEED

parser = argparse.ArgumentParser(description="""Run the NN tagger""")
#parser.add_argument("--train", help="train data")  # allow multiple train files, each asociated with a task = position in the list
# parser.add_argument("--dev", help="dev file(s)", required=False)
#parser.add_argument("--test", help="test file(s)", required=False)
parser.add_argument("--lang", help="lang prefix", required=False)
parser.add_argument("--dynet-seed", help="random seed for dynet (needs to be first argument!)", required=False, type=int)
parser.add_argument("--dynet-mem", help="memory for dynet (needs to be first argument!)", required=False, type=int)
args = parser.parse_args()


BASE="/projdata/alpage2/hmartine/data/ud1.3/orgtok/goldpos/"
LEXBASE="/projdata/alpage2/hmartine/data/lex/"
#BASE="/Users/sagot/Documents/al/al_tagger/trunk/ud13/"
#LEXBASE="/Users/sagot/Documents/al/al_tagger/trunk/lex/"
seed=113 # assume we pass this to script
#train_data = args.train #"/Users/bplank/corpora/pos/ud1.3/orgtok/goldpos/da-ud-dev.conllu"
#dev_data = args.dev #"/Users/bplank/corpora/pos/ud1.3/orgtok/goldpos/da-ud-test.conllu"
#test_data = args.test #"/Users/bplank/corpora/pos/ud1.3/orgtok/goldpos/da-ud-test.conllu"

train_data = BASE+args.lang + "-ud-train.conllu"
dev_data = BASE + args.lang + "-ud-dev.conllu"
test_data = BASE + args.lang + "-ud-test.conllu"
lexfile = LEXBASE + args.lang + ".lex"

in_dim=64
h_dim=100
c_in_dim=100
h_layers=1
iters=20
trainer="sgd"
tagger = SimpleBiltyTagger(in_dim, h_dim,c_in_dim,h_layers,embeds_file=None,lex_file=lexfile,coarse_lex=1)
train_X, train_Y = tagger.get_train_data(train_data,tagger.w2i)
dev_X, dev_Y = tagger.get_data_as_indices(dev_data)
tagger.fit(train_X, train_Y, iters, trainer,seed=seed,dev_X=dev_X,dev_Y=dev_Y)
test_X, test_Y = tagger.get_data_as_indices(test_data)
correct, total = tagger.evaluate(test_X, test_Y)
print("test accuracy",correct, total, correct/total)
