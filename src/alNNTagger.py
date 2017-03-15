# bidirectional model
# https://github.com/fchollet/keras/issues/1629
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Input, TimeDistributed, Dropout, TimeDistributedDense, merge
from keras.layers.recurrent import Recurrent
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import argparse
import numpy as np
import sys

w2i_dict = dict()  # default value for unknown words
l2i_dict = dict()

max_features = 10000
max_length = 50
embedding_dim = 64
batch_size = 128
epochs = 2


def read_tab_sep(file_name):
    current_words = []
    current_tags = []

    for line in open(file_name).readlines():
        line = line.strip()

        if line:
            if len(line.split("\t")) != 2:
                if len(line.split("\t")) == 1:
                    raise IOError("Issue with input file - doesn't have a tag or token?")
                else:
                    print("erroneous line: {} (line number: {}) ".format(line, i))
                    exit()
            else:
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

def read_lexicon(infile, lowercase=False):
    # TODO: read lex in, for each word index, generate a list of binaries size |Traits|
    # TODO: expand input of the NN to accept lexicon entry
    return None


def read_embeddings(infile):
    # TODO: read embeddings
    return None


def main():
    parser = argparse.ArgumentParser(description="""toy LSTM""")
    parser.add_argument("--train",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--test",default="corpus/pos_ud_en_test.2col")
    parser.add_argument("--dev",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--lexicon",default="lex/pos_ud_en_dev.conll")
    parser.add_argument("--max_features",default=2000, )



    args = parser.parse_args()
    trainfile = args.train
    testfile = args.test

    for wordseq, labelseq in list(read_tab_sep(trainfile)):
        [string2index(w,l2i_dict) for w in labelseq]
    nb_tags = len(l2i_dict.keys())
    
    train_X = []
    train_Y = []
    train_Y_old = []
    for wordseq, labelseq in list(read_tab_sep(trainfile)):
        train_x = [string2index(w,w2i_dict,max_dict_size=1999) for w in wordseq]
        train_y = [string2index(w,l2i_dict) for w in labelseq] #,nb_tags
        train_X.append(train_x)
        train_Y.append(train_y)
    train_X = pad_sequences(train_X,maxlen=max_length)
    train_Y = pad_sequences(train_Y,maxlen=max_length)
    train_Y = np.array([np_utils.to_categorical(seq, nb_tags) for seq in train_Y])
   

    test_X = []
    test_Y = []
    for wordseq, labelseq in read_tab_sep(testfile):
        test_x = [string2index(w, w2i_dict, False,max_dict_size=1999) for w in wordseq]
        test_y = [string2index(w, l2i_dict, False) for w in labelseq]
        test_X.append(test_x)
        test_Y.append(test_y)
    test_X = pad_sequences(test_X,maxlen=max_length)
    test_Y = pad_sequences(test_Y,maxlen=max_length)
    test_Y = np.array([np_utils.to_categorical(seq, nb_tags) for seq in test_Y])

    print ("data reading done")

    sequence = Input(shape=(max_length,),dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_length, mask_zero=True)(sequence)
    fwd_lstm = LSTM(output_dim=embedding_dim, return_sequences=True)(embedded)
    bwd_lstm = LSTM(output_dim=embedding_dim, go_backwards=True, return_sequences=True)(embedded)
    merged = merge([fwd_lstm, bwd_lstm], mode='concat', concat_axis=-1)
    droppedout = Dropout(0.2)(merged)
    densed = TimeDistributed(Dense(output_dim=nb_tags))(droppedout)
    output = Activation('softmax')(densed)
    model = Model(input=sequence, output=output)
    
    print ("model building done")
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'],sample_weight_mode='temporal')
    print ("about to fit")
    model.fit(train_X, train_Y,batch_size=batch_size, nb_epoch=epochs, validation_data=(test_X, test_Y))


if __name__ == "__main__":
    main()
