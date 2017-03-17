# bidirectional model
# https://github.com/fchollet/keras/issues/1629
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Input, TimeDistributed, Dropout, merge
#from keras.layers.recurrent import Recurrent
from keras.layers.merge import concatenate
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



def string2index(word,dictindex,update=False,max_dict_size=0):
    if update:
        dictindex[""] = 0
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
    return len(tag_index),L


def read_annotated_file(infile,w2i_dict,l2i_dict,max_features,max_sequence_length,update_w2i=False,update_l2i=False):
    test_X = []
    test_Y = []
    nb_tags = len(l2i_dict.keys())

    for wordseq, labelseq in read_tab_sep(infile):
        test_x = [string2index(w, w2i_dict, update_w2i,max_dict_size=max_features) for w in wordseq]
        test_y = [string2index(w, l2i_dict, update_l2i) for w in labelseq]
        test_X.append(test_x)
        test_Y.append(test_y)
    test_X = pad_sequences(test_X,maxlen=max_sequence_length)
    test_Y = pad_sequences(test_Y,maxlen=max_sequence_length)
    test_Y = np.array([np_utils.to_categorical(seq, nb_tags) for seq in test_Y])
    return test_X,test_Y

def create_lexicon_matrix(X,lex,max_sequence_length,lextag_nb):
    # prepare lexical matrix, aligned word-wise

    Xlexcl = np.zeros((X.shape[0],max_sequence_length,lextag_nb))
    for i,sentence in enumerate(X):
        for j,word in enumerate(sentence):
            if word in lex:
                Xlexcl[i][j] = lex[word]
    return Xlexcl

def create_cemb_matrix(X,i2w_dict,c2i_dict,max_sequence_length,max_token_length):
    # prepare cemb matrix, aligned word-wise

    Xcemb = np.zeros((X.shape[0],max_sequence_length,max_token_length))
    for i,sentence in enumerate(X):
        for j,word in enumerate(sentence):
            for k,char in enumerate(i2w_dict[word]):
                if k < max_token_length:
                    Xcemb[i][j][k] = c2i_dict[char]
                else:
                    break
    return Xcemb

def main():
    parser = argparse.ArgumentParser(description="""toy LSTM""")
    parser.add_argument("--train",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--test",default="corpus/pos_ud_en_test.2col")
    parser.add_argument("--dev",default="corpus/pos_ud_en_dev.2col")
    parser.add_argument("--lexicon",default="lex/en_lexicon_short.ftl")
    parser.add_argument("--embeddings",default=None)
    parser.add_argument("--cemb_layer_size",type=int,default=64) # 0 = do not use character embeddings
    parser.add_argument("--max_features",type=int,default=5000 )
    parser.add_argument("--max_sequence_length",type=int,default=50 )
    parser.add_argument("--max_token_length",type=int,default=25 )
    parser.add_argument("--embedding_dim",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--batch_size",type=int,default=128)

    args = parser.parse_args()
    trainfile = args.train
    testfile = args.test


    w2i_dict = dict()  # default value for unknown words
    l2i_dict = dict()
    c2i_dict = dict()

    # initialisation of the label (i.e. tag) dictionary
    for wordseq, labelseq in list(read_tab_sep(trainfile)):
        [string2index(w,l2i_dict,update=True) for w in labelseq]
    nb_tags = len(l2i_dict.keys())

    train_X,train_Y=read_annotated_file(trainfile,w2i_dict,l2i_dict,args.max_features,args.max_sequence_length,update_w2i=True,update_l2i=True)
    test_X,test_Y=read_annotated_file(testfile,w2i_dict,l2i_dict,args.max_features,args.max_sequence_length)
    
    observed_n_feats = args.max_features + 1

    embedding_matrix = None
    if args.embeddings:
        embedding_dict =read_embed_file(args.embeddings)
        embedding_matrix=create_embedding_matrix(embedding_dict,w2i_dict,args.embedding_dim)

    lexicon = None
    nb_lexclasses = 0
    if args.lexicon:
        nb_lexclasses,lexicon = read_lexicon(args.lexicon)
        train_Xlexcl = create_lexicon_matrix(train_X,lexicon,args.max_sequence_length,nb_lexclasses)
        test_Xlexcl = create_lexicon_matrix(test_X,lexicon,args.max_sequence_length,nb_lexclasses)

    if args.cemb_layer_size > 0:
        # initialisation of the character dictionary (for character embeddings)
        [string2index(char,c2i_dict,update=True) for char in sorted(set("".join(list(w2i_dict.keys()))))]
        # creating character embedding data
        i2w_dict = dict([(i,w) for (w,i) in w2i_dict.items()])
        i2w_dict[len(w2i_dict.keys())+1] = "" # unknown word
        train_Xcemb = create_cemb_matrix(train_X,i2w_dict,c2i_dict,args.max_sequence_length,args.max_token_length)
        test_Xcemb = create_cemb_matrix(test_X,i2w_dict,c2i_dict,args.max_sequence_length,args.max_token_length)
        
    print ("data reading done")

    sequence = Input(shape=(args.max_sequence_length,),dtype='int32')

    #TODO review masked zero!
    """ mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out.
    This is useful when using recurrent layers which may take variable length input.
    If this is True then all subsequent layers in the model need to support masking or an exception will be raised.
    If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary
    (input_dim should equal |vocabulary| + 2)."""
    
    embedded = Embedding(input_dim=observed_n_feats, output_dim=args.embedding_dim,
                            input_length=args.max_sequence_length,
                            weights=embedding_matrix, # =None if no embeddings provided
                            mask_zero=False)(sequence)

    to_be_concatenated = [embedded]
    inputs = [sequence]
    train_data = [train_X]
    test_data = [test_X]
    lstminput_width = args.embedding_dim

    if lexicon:
        lexclassed = Input(shape=(args.max_sequence_length,nb_lexclasses),dtype='float32',name="charmodelinput")
        to_be_concatenated.append(lexclassed)
        inputs.append(lexclassed)
        train_data.append(train_Xlexcl)
        test_data.append(test_Xlexcl)
        lstminput_width += nb_lexclasses

    if args.cemb_layer_size > 0:
        charsequence = Input(shape=(args.max_token_length,),dtype='int32')
        charembedded = Embedding(input_dim=len(c2i_dict.keys()), output_dim=args.cemb_layer_size,
                             input_length=args.max_token_length,
                             mask_zero=False)(charsequence)
        fwd_clstm = LSTM(units=args.cemb_layer_size, return_sequences=False)(charembedded)
        bwd_clstm = LSTM(units=args.cemb_layer_size, go_backwards=True, return_sequences=False)(charembedded)
        charmerged = concatenate([fwd_clstm, bwd_clstm], axis=-1)
        chardroppedout = Dropout(0.2)(charmerged)
        charembedded = Dense(units=args.cemb_layer_size)(chardroppedout)
        charlevelmodel = Model(inputs=[charsequence],outputs=[charembedded])

        charlevelmodelinput = Input(shape=(args.max_sequence_length,args.max_token_length),dtype='int32',name="charmodelcallinginput")
        charlevelmodeloutput = TimeDistributed(charlevelmodel)(charlevelmodelinput)

        to_be_concatenated.append(charlevelmodeloutput)
        inputs.append(charlevelmodelinput)
        train_data.append(train_Xcemb)
        test_data.append(test_Xcemb)
        lstminput_width += args.cemb_layer_size

    if len(to_be_concatenated) == 1:
        lstminput = embedded
    else:
        lstminput = concatenate(to_be_concatenated, axis=-1)

    fwd_lstm = LSTM(units=lstminput_width, return_sequences=True)(lstminput)
    bwd_lstm = LSTM(units=lstminput_width, go_backwards=True, return_sequences=True)(lstminput)
    merged = concatenate([fwd_lstm, bwd_lstm], axis=-1)
    droppedout = Dropout(0.2)(merged)
    densed = TimeDistributed(Dense(units=nb_tags))(droppedout)
    output = Activation('softmax')(densed)

    model = Model(inputs=inputs, outputs=output)
    
    print ("model building done")
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'],sample_weight_mode='temporal')
    print ("about to fit")
    model.fit(train_data, train_Y,batch_size=args.batch_size, epochs=args.epochs, validation_data=(test_data, test_Y))


if __name__ == "__main__":
    main()
