#import math
import random
import sys
import numpy as np
#import json
#import pickle
#from collections import OrderedDict
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
from sklearn import svm
from sklearn import metrics
from yellowbrick.text import TSNEVisualizer
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
from keras.layers import LSTM, Dense, Embedding, Input, Layer
from keras.models import Model



from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

#sys.stdout = open("attention.txt", "w")


def get_model(model_type, *args, **kwargs):
    if model_type == "lstm":
        return _lstm_based_model(*args, **kwargs)
    if model_type == "attention":
        return _attention_based_model(*args, **kwargs)
    raise NotImplementedError


def _lstm_based_model(maxlen, vocab_size, emb_size, hidden_size=32, mask_zero=True):
    inp = Input(shape=[maxlen])
    emb = Embedding(vocab_size, emb_size, mask_zero=mask_zero,)
    x = emb(inp)
    x = LSTM(hidden_size)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    return model


def _attention_based_model(
    maxlen,
    vocab_size,
    emb_size,
    hidden_size=32,
    attention_hs=16,
    mask_zero=True,
    return_attn_weights=False,
):
    inp = Input(shape=[maxlen])
    emb = Embedding(vocab_size, emb_size, mask_zero=mask_zero,)
    x = emb(inp)
    x, hs, cs = LSTM(hidden_size, return_sequences=True, return_state=True)(x)
    x, weights = BahdanauAttention(attention_hs)(hs, x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    if return_attn_weights:
        model_attention = Model(inputs=inp, outputs=weights)
        return model, model_attention
    return model


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # TODO: Add masking
    def call(self, query, values):
        # query : [batch_size, hidden_size]
        # values: [batch_size, maxlen, hidden_size]

        # (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # (batch_size, maxlen, units) + (batch_size, 1, units) = (batch_size, maxlen, units)
        score = self.W1(values) + self.W2(hidden_with_time_axis)

        # (batch_size, maxlen, 1)
        score = self.V(tf.nn.tanh(score))

        # attention_weights shape == (batch_size, maxlen, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights




We modified and used the code of Öztürk et. al.(2018) for integer encoding.


filepth = 'C:/Users/Nibba/PycharmProjects/DTA/histoneproject/'
reader_histone_interactions = pd.read_csv(filepth+"histone-reader-positive.csv")
re = reader_histone_interactions["UniProtID"]
his = reader_histone_interactions ["Substrate"]

rdata = pd.read_csv(filepth+"readers.csv")
hisdata = pd.read_csv(filepth+"histones.csv")

rid = rdata["uniprotid"]
subs = hisdata["Substrate"]
reader_seq = rdata["sequence"]

#
#  Define CHARSET, CHARLEN


CHARPROTSET = {"A": 64, "C": 65, "B": 66, "E": 67, "D": 68, "G": 69,
               "F": 70, "I": 71, "H": 72, "K": 73, "M": 74, "L": 75,
               "O": 76, "N": 77, "Q": 78, "P": 80, "S": 81, "R": 82,
               "U": 83, "T": 84, "W": 85,
               "V": 86, "Y": 87, "X": 88,
               "Z": 89}

CHARPROTLEN = 25



CHARISOSMISET = {"C": 6, "(": 7, ")": 10, "=": 11, "+": 13, "-": 15, "@": 19, "[": 21,
                 "]": 22,"O": 24,"N": 25,"H": 26, "3": 28}

CHARISOSMILEN = 13


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ##

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  # .tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]
    return X  # .tolist()


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        try:
            X[i] = smi_ch_ind[ch]
        except:
            continue

    return X  # .tolist()


## ######################## ##
#
#  DATASET Class
#
## ######################## ##
# works for large dataset
class DataSet(object):
    def __init__(self, fpath, setting_no, seqlen, smilen,fseqlen, need_shuffle=False):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.FSEQLEN = fseqlen
        # self.NCLASSES = n_classes
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET  ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting_no
        self.fpath = fpath
        # read raw file
        # self._raw = self.read_sets( FLAGS)

        # iteration flags
        # self._num_data = len(self._raw)


    def parse_data(self, with_label=True):
        print("Read %s start" % self.fpath)
        readerdata = pd.read_csv(filepth+"readers.csv")
        histonedata = pd.read_csv(filepth+"histones.csv")

        rid = readerdata["uniprotid"]
        rseq = readerdata["sequence"]

        hindex = histonedata["index"]
        rindex = readerdata["index"]
        subs = histonedata["Substrate"]
        aatype = histonedata["AA"]
        htype = histonedata["Histone"]
        hmarkpos = histonedata["Mark position"]
        hflankingseq = histonedata["Flanking sequence"]
        hsmiles = histonedata["SMILES"]

        XR = []
        XH = []
        XFs = []
        XA = []
        XHt = []
        if with_label:
            for d in hsmiles:
                # print("ligand smiles:", d )
                # print("ligands[d]",ligands[d])

                XH.append(label_smiles(d, self.SMILEN, self.charsmiset))
            # print("self.smilen",self.SMILEN)
            # rint("self.charsmiset",self.charsmiset)
            for t in rseq:
                # print("protein seq:",t)
                # print("protein:",proteins[t])
                #targets[t] = proteins[t]
                XR.append(label_sequence(t, self.SEQLEN, self.charseqset))
            # print("self.seqlen", self.SEQLEN)
            # print("self.charseqset:",self.charseqset)
            #targetsdf = pd.DataFrame.from_dict(targets, orient='index')
            # targetsdf.to_csv("targetsdf.csv", index=False)
            for mseq in hflankingseq:
                XFs.append(label_sequence(mseq, self.FSEQLEN, self.charseqset))
            for aa in aatype:
                XA.append(label_sequence(aa, 1, self.charseqset))
            for ht in htype:
                if ht == "H3":
                    XHt.append([31])
                elif ht == "H4":
                    XHt.append([32])
                else:
                    print("error")
        '''else:
            for d in ligands.keys():
                XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))
                # print("ligands[d]",ligands[d])
                # print("one hot smiles:",one_hot_smiles(ligands[d]))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))'''
        # print("XD: ", XD)
        # print("\nXT:", XT)
        # np.savetxt("Y.csv", Y, delimiter=",")
        # print(Y.shape)
        # print("print XD:",XD)
        #print("XD shape:", [len(XD), len(XD[0]),len(XD[0][0])])
        #print("XT shape", [len(XT), len(XT[0]),len(XT[0][0])])
        return XR, XH, XFs, XA, XHt

dataset = DataSet( fpath = '',
                  setting_no = 1, ##BUNU ARGS A EKLE
                  seqlen = 1550,
                  smilen = 33,
                    fseqlen= 7,
                  need_shuffle = False )
# set character set size
#FLAGS.charseqset_size = dataset.charseqset_size
#FLAGS.charsmiset_size = dataset.charsmiset_size

XR, XH, XFs, XA, XHt = dataset.parse_data()
print(XR[0])
print(XH[0])
print(XFs[0])
print(XA[0])
print(XHt[0])

print(len(XR[0]))
print(len(XH[0]))
print(len(XFs[0]))
print(len(XA[0]))
print(len(XHt[0]))
# # print(type(XD),len(XD[0]),len(XT[0]))
# mylist=[]
# # print(Y[0,0])
# # print(Y[0,0].size)
# print(Y[0,0])
# print("type of Y:",type(Y[0,0]))
# print("Type of xt:,",type(XT[0]))
# mylist.append(np.concatenate((XT[0],XD[0])))
# print(XD[0].shape+XT[0].shape)
# print(mylist)
#
# print(len(mylist[0]))
histone_data = []
for k,h in enumerate(XH):
    histone_data.append(np.concatenate((XH[k], XFs[k], XHt[k], XA[k])))

"""pca1 = PCA(n_components=30)
pca2 = PCA(n_components=30)
pca1.fit(XR)
pca2.fit(histone_data)

R_embedded = pca1.transform(XR)
H_embedded = pca2.transform(histone_data)

print(R_embedded.shape)
print(H_embedded.shape)"""

'''tsne0 = TSNEVisualizer(decompose='pca', decompose_by= 107,title="TSNE Projection of 107 Readers")
tsne0.fit(XR, rid)
tsne0.show()

tsne1 = TSNEVisualizer(decompose='pca', decompose_by= 42,title="TSNE Projection of 57 Histones")
tsne1.fit(histone_data, subs)
tsne1.show()
'''


mydataset1=[]
mylabels1=[]
counter = []
c = 0
for i, readers in enumerate(XR):
    for j, histones in enumerate(XH):
        mydataset1.append(np.concatenate((XR[i],histone_data[j])))
        mylabels1.append(-1)
        for r1, h1 in zip(re,his):
            if r1 == rid[i] and h1 == subs[j]:
                #print(r1,h1)
                counter.append(c)

        c = c+1
for ind1 in counter:
    mylabels1[ind1] = 1
mydataset2 = mydataset1
print(np.shape(mydataset2))
print(np.shape(mylabels1))
mylabels1 = np.reshape(mylabels1, (6099,1))
mydataset2 = np.append(mydataset2,mylabels1,axis=1)

mydata = np.array(mydataset2)
print(np.shape(mydata))
print(mydata[1])


#print(np.shape(mydata))
#print(np.shape(mydataset2))
#print(len(mydata[1]))
#print(len(mydataset2[1]))
#print(mydata[-1])
#print(len(mydata[-1]))
#print(np.shape(mydata))

#np.random.shuffle(mydata)

#print(np.shape(mydata))


new_data = mydata[:,:-1]
new_labels = mydata[:,-1]

print(np.shape(new_data))
print(np.shape(new_labels))
print(new_labels)
#print(new_data)

datax = []
testx = []
testl = []
negL = []
c = 0
for label in new_labels:
    if label == -1:
        datax.append(new_data[c])
        negL.append(0)
    c = c+1

c = 0
for label in mylabels1:
    if label == 1:
        testx.append(mydataset1[c])
        testl.append(1)
    c = c+1
sneg = datax
np.random.shuffle(sneg)
newneg = sneg[:6750]
newnegL = negL[:6750]

testl = np.reshape(testl, (270, 1))
testx = np.array(testx)
newposwithlabels = np.append(testx, testl, axis=1)
np.random.shuffle(newposwithlabels)
posfortest = newposwithlabels[240:]

valset = newposwithlabels[220:240]
newposwithlabels = newposwithlabels[:220]
#cp_filepath = 'my_best_model.hdf5'

cppath = 'C:/Users/Nibba/PycharmProjects/DTA/histoneproject/tmp/'
newneg = np.array(newneg)
for i in range(26):
    newnewneg = newneg[220*i:220*(i+1)]
    newnewnegL = newnegL[220*i:220*(i+1)]
    checkpoint_filepath = cppath+str(i+1)+'/'
    newnewnegL = np.reshape(newnewnegL,(220,1))


    print(np.shape(newneg))

    print(np.shape(newnegL))

    newnegwithlabels = np.append(newnewneg,newnewnegL,axis=1)




    print(np.shape(newnegwithlabels))
    print(np.shape(newposwithlabels))
    alldata = np.append(newnegwithlabels,newposwithlabels,axis=0)
    print(alldata[1])
    print(np.shape(alldata))

    np.random.shuffle(alldata)
    print("all data shape",np.shape(alldata))
    xtrain = alldata
    #xtest = alldata[540:]

    print("unedited")
    print(xtrain[1])
    print(xtrain[1].shape)
    print(xtrain[1,:-1])
    print(xtrain[1,:-1].shape)

    y_train = xtrain[:,-1]
    #y_test = xtest[:,-1]

    x_train = xtrain[:,:-1]
    #x_test = xtest[:,:-1]
    print(np.shape(x_train))
    print("sample train")
    print(x_train[0,:-1])
    print(x_train[1, :-1])

    x_test = posfortest[:,:-1]
    y_test = posfortest[:,-1]
    x_val = valset[:,:-1]
    y_val = valset[:,-1]
    #print(np.shape(x_test))
    es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor = 'val_mse',
        mode = 'min',
        save_weights_only=True,
        save_best_only=True
    )

    max_words = 1592
    vocab_size = 90
    emb_size = 16
    batch_size = 64
    epochs = 3
    model_d, model_attention = get_model(
        "attention",
        max_words,
        vocab_size,
        emb_size,
        attention_hs=16,
        return_attn_weights=True,
    )
    model_d.summary()


    for idx,value in x_test:
        plot_attention(attn_weights[idx], x_test[idx])
        plt.show()

    test_pred = model_d.predict(x_test, batch_size=64, verbose=1)
    #y_pred_bool = np.argmax(y_pred, axis=0)
    test_pred_bool = []
    """




    print("New positives found in the iteration number"+str(i)+"=", count1)

    print(classification_report(y_train, y_pred,zero_division=0))
    #print(precision_recall_fscore_support(y_train,y_pred_bool, zero_division=0))
    print("Test accuracy\n")
    #print(classification_report(y_test,test_pred_bool,zero_division=0))

#sys.stdout.close()

exit(0)

postest = []
poslabel = []
c = 0
for label in y_test:
    if label == 1:
        postest.append(x_test[c])
        poslabel.append(1)
    c = c+1

postest = np.array(postest)
poslabel = np.array(poslabel)

results = model_d.evaluate(postest,poslabel)