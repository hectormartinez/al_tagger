# bidirectional model
# https://github.com/fchollet/keras/issues/1629
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Input, TimeDistributed, Dropout, TimeDistributedDense, merge
from keras.layers.recurrent import Recurrent
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import pandas as pd
import argparse
import numpy as np
import sys



def read_embed_file(infile,sep=" ",lower=False):
    # management for OOV? Remember there is an "_UNK" in polyglot
    E = {}
    for line in open(infile).readlines():
        fields = line.strip().split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        E[word] = vec
    return E


def create_embedding_matrix(embed_dict,w2i_dict,embedding_dim):
    # prepare embedding matrix, aligned index-wise with numeric word indices
    # The following +2 is a result of the patching we do with max_features
    # If the max index is n_feats+1, then the size of the layer is n_feats+2

    num_words = len(w2i_dict.keys())+2
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in w2i_dict.items():
        embedding_vector = embed_dict.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return [embedding_matrix]

def read_tab_sep(file_name):
    current_words = []
    current_tags = []
    for line in open(file_name).readlines():
        line = line.strip()
        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            if current_words:
                yield (current_words, current_tags)
            current_words = []
            current_tags = []
    if current_tags != []:
        yield (current_words, current_tags)



def string2index(word,dictindex,update=True,max_dict_size=0):
    if word in dictindex:
        return dictindex[word]
    elif update:
        if max_dict_size > 0 and len(dictindex.keys()) == max_dict_size - 1:
            return max_dict_size
        else:
            dictindex[word] = len(dictindex.keys())
            return dictindex[word]
    else:
        if max_dict_size > 0:
            return max_dict_size
        else:
            return len(dictindex.keys())+1

def read_lexicon(infile):
    # TODO: lowercase option missing
    L = dict()
    frame = pd.read_csv(infile,'\t',names=["form","tag","lemma"])
    tag_index = sorted([str(x) for x in set(list((frame.tag)))])
    for form in set(list(frame.form)):
        tags_for_words = set(list(frame[frame.form == form].tag))
        L[form] = [1 if tag_index[i] in tags_for_words else 0 for i in range(len(tag_index))]
    return L


def read_annotated_file(infile,w2i_dict,l2i_dict,max_features,max_sequence_length,update_l2i=False):
    test_X = []
    test_Y = []
    nb_tags = len(l2i_dict.keys())

    for wordseq, labelseq in read_tab_sep(infile):
        test_x = [string2index(w, w2i_dict, update_l2i,max_dict_size=max_features) for w in wordseq]
        test_y = [string2index(w, l2i_dict, update_l2i) for w in labelseq]
        test_X.append(test_x)
        test_Y.append(test_y)
    test_X = pad_sequences(test_X,maxlen=max_sequence_length)
    test_Y = pad_sequences(test_Y,maxlen=max_sequence_length)
    test_Y = np.array([np_utils.to_categorical(seq, nb_tags) for seq in test_Y])
    return test_X,test_Y

def main():
    parser = argparse.ArgumentParser(description="""toy LSTM""")
    parser.add_argument("--train",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--test",default="corpus/pos_ud_en_test.2col")
    parser.add_argument("--dev",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--lexicon",default="lex/en_lexicon_short.ftl")
    parser.add_argument("--embeddings",default=None)
    parser.add_argument("--max_features",type=int,default=5000 )
    parser.add_argument("--max_sequence_length",type=int,default=50 )
    parser.add_argument("--embedding_dim",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=2 )
    parser.add_argument("--batch_size",type=int,default=128)

    args = parser.parse_args()
    trainfile = args.train
    testfile = args.test


    w2i_dict = dict()  # default value for unknown words
    l2i_dict = dict()

    for wordseq, labelseq in list(read_tab_sep(trainfile)):
        [string2index(w,l2i_dict) for w in labelseq]
    nb_tags = len(l2i_dict.keys())

    train_X,train_Y=read_annotated_file(trainfile,w2i_dict,l2i_dict,args.max_features,args.max_sequence_length)
    test_X,test_Y=read_annotated_file(testfile,w2i_dict,l2i_dict,args.max_features,args.max_sequence_length,update_l2i=False)


    observed_n_feats = args.max_features + 1

    if args.embeddings:
        embedding_dict =read_embed_file(args.embeddings)
        embedding_matrix=create_embedding_matrix(embedding_dict,w2i_dict,args.embedding_dim)

    lexicon = None
    if args.lexicon:
        lexicon = read_lexicon(args.lexicon)

    print ("data reading done")

    sequence = Input(shape=(args.max_sequence_length,),dtype='int32')


    #TODO review masked zero!
    """ mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out.
    This is useful when using recurrent layers which may take variable length input.
    If this is True then all subsequent layers in the model need to support masking or an exception will be raised.
    If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary
    (input_dim should equal |vocabulary| + 2)."""

    print (embedding_matrix.shape)
    
    if args.embeddings:
        embedded = Embedding(input_dim=observed_n_feats, output_dim=args.embedding_dim,
                         input_length=args.max_sequence_length, weights=embedding_matrix,mask_zero=False)(sequence)
    else:
        embedded = Embedding(input_dim=observed_n_feats, output_dim=args.embedding_dim,
                             input_length=args.max_sequence_length, mask_zero=False)(sequence)

    fwd_lstm = LSTM(output_dim=args.embedding_dim, return_sequences=True)(embedded)
    bwd_lstm = LSTM(output_dim=args.embedding_dim, go_backwards=True, return_sequences=True)(embedded)
    merged = merge([fwd_lstm, bwd_lstm], mode='concat', concat_axis=-1)
    droppedout = Dropout(0.2)(merged)
    densed = TimeDistributed(Dense(output_dim=nb_tags))(droppedout)
    output = Activation('softmax')(densed)
    model = Model(input=sequence, output=output)
    
    print ("model building done")
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'],sample_weight_mode='temporal')
    print ("about to fit")
    model.fit(train_X, train_Y,batch_size=args.batch_size, nb_epoch=args.epochs, validation_data=(test_X, test_Y))


if __name__ == "__main__":
    main()
