# Keras==1.0.6
# from https://gist.githubusercontent.com/dirko/1d596ca757a541da96ac3caa6f291229/raw/319ef73c357b263b447117537d5237e6b39d04be/keras_bidirectional_tagger.py
from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Merge, Layer
from keras.backend import tf
from lambdawithmask import Lambda as MaskLambda
from sklearn.metrics import confusion_matrix, accuracy_score


def load_data_from_2col_file(infile):
    all_x = []
    point = []
    raw = open(infile, 'r').readlines()

    for line in raw:
        stripped_line = line.strip().split('\t')
        point.append(stripped_line)
        if line == '\n':
            all_x.append(point[:-1])
            point = []
    all_x = all_x[:-1]
    lengths = [len(x) for x in all_x]
    X = [[c[0] for c in x] for x in all_x]
    Y = [[c[1] for c in y] for y in all_x]
    all_text = [c for x in X for c in x]
    return X,Y,all_text,lengths


def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def prep_data():

    #TODO lots of replicated code, but if it works, it works
    all_text = []
    lengths = []
    train_X, train_Y, train_all_text,train_lengths = load_data_from_2col_file('../corpus/pos_ud_en_train.2col')
    dev_X, dev_Y, dev_all_text,dev_lengths = load_data_from_2col_file('../corpus/pos_ud_en_dev.2col')
    test_X, test_Y, test_all_text,test_lengths = load_data_from_2col_file('../corpus/pos_ud_en_test.2col')

    all_text.extend(test_all_text)
    all_text.extend(train_all_text)
    all_text.extend(dev_all_text)

    lengths.extend(train_lengths)
    lengths.extend(test_lengths)
    lengths.extend(dev_lengths)

    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = list(set([c for x in train_Y+test_Y+dev_Y for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    print('Input sequence length range: ', max(lengths), min(lengths))

    #maxlen = max([len(x) for x in X])
    print('Maximum sequence length:', max(lengths))

    maxlen = max(lengths)
    max_label = max(label2ind.values()) + 1



    train_X_enc = [[word2ind[c] for c in x] for x in train_X]

    train_y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in train_Y]
    train_y_enc = [[encode(c, max_label) for c in ey] for ey in train_y_enc]

    train_X_enc = pad_sequences(train_X_enc, maxlen=maxlen)
    train_y_enc = pad_sequences(train_y_enc, maxlen=maxlen)

    test_X_enc = [[word2ind[c] for c in x] for x in test_X]

    test_y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in test_Y]
    test_y_enc = [[encode(c, max_label) for c in ey] for ey in test_y_enc]

    test_X_enc = pad_sequences(test_X_enc, maxlen=maxlen)
    test_y_enc = pad_sequences(test_y_enc, maxlen=maxlen)

    return train_X_enc, test_X_enc, train_y_enc, test_y_enc,word2ind,label2ind,maxlen

def reverse_func(x, mask=None):
    return tf.reverse(x, [False, True, False])


def main():

    X_train, X_test, y_train, y_test,word2ind,label2ind,maxlen = prep_data()

    print ('Training and testing tensor shapes:')
    print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    max_features = len(word2ind)
    embedding_size = 128
    hidden_size = 32
    out_size = len(label2ind) + 1

    # model_forward = Sequential()
    # model_forward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
    # model_forward.add(LSTM(hidden_size, return_sequences=True))

    # model_backward = Sequential()
    # model_backward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
    # model_backward.add(LSTM(hidden_size, return_sequences=True))
    # model_backward.add(MaskLambda(function=reverse_func, mask_function=reverse_func))

    model = Sequential()

    model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
#    model.add(Merge([model_forward, model_backward], mode='concat'))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    batch_size = 32
    model.fit([X_train], y_train, batch_size=batch_size, epochs=100,
              validation_data=([X_test], y_test))
    raw_test_score = model.evaluate([X_test], y_test, batch_size=batch_size)
    print('Raw test score:', raw_test_score)


    pr = model.predict_classes([X_train])
    yh = y_train.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Training accuracy:', accuracy_score(fyh, fpr))
    print('Training confusion matrix:')
    print(confusion_matrix(fyh, fpr))

    pr = model.predict_classes([X_test])
    yh = y_test.argmax(2)
    fyh, fpr = score(yh, pr)
    print ('Testing accuracy:', accuracy_score(fyh, fpr))
    print ('Testing confusion matrix:')
    print (confusion_matrix(fyh, fpr))

if __name__ == "__main__":
    main()
