# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from deepctr.models import DIN

import sys
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_fixlen_feature_names,get_varlen_feature_names
epochs = 5
DIN_SESS_MAX_LEN = 100

use_gpu = sys.argv[1]
ebs = sys.argv[2]
dnn_dropout = sys.argv[3]
print(use_gpu,"----",ebs,"----",dnn_dropout)

os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

date_list = ["20190610","20190611","20190612","20190613","20190614","20190615","20190616","20190617","20190618","20190619"]
test_date = "20190620"
if __name__ == "__main__":
    SESS_MAX_LEN = DIN_SESS_MAX_LEN
    fd = pd.read_pickle('../model_input/din_fd.pkl')
    print(fd)
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 1024
    sess_feature = ['item_id']
    # def auc(y_true,y_pred):
    #   return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    EMBEDDING_SIZE = int(ebs)
    if EMBEDDING_SIZE == 0:
      EMBEDDING_SIZE = "auto"

    model = DIN(fd, sess_feature, embedding_size=EMBEDDING_SIZE, dnn_dropout=float(dnn_dropout), att_activation='dice',
                att_weight_normalization=False, hist_len_max=sess_len_max, dnn_hidden_units=(200, 80),
                att_hidden_size=(64, 16,))
    model.compile('adagrad', 'binary_crossentropy',
                    metrics=['binary_crossentropy'])
    model_dir = "../model_dir_"+str(EMBEDDING_SIZE)
    if not os.path.exists(model_dir):
      os.mkdir(model_dir)
    if os.path.exists(model_dir+'/ckpt.h5'):
      model.load_weights(model_dir+'/ckpt.h5')

    """
    test_input_pos = pd.read_pickle(
          '../model_input/din_input_'+test_date+'.pkl')
    test_input_neg = pd.read_pickle(
          '../model_input/din_input_'+test_date+'_neg.pkl')
    test_input = []
    for i in range(len(test_input_pos)):
      feature_input = np.concatenate([test_input_pos[i],test_input_neg[i]],axis=0)
      test_input.append(feature_input)
      # model_input = np.concatenate([model_input_pos,model_input_neg],axis=1)
    test_label_pos = pd.read_pickle('../model_input/din_label_'+test_date+'.pkl')
    test_label_neg = pd.read_pickle('../model_input/din_label_'+test_date+'_neg.pkl')
    test_label = np.concatenate([test_label_pos,test_label_neg],axis=-1)

    del test_input_pos
    del test_input_neg
    del test_label_pos
    del test_label_neg

    pred_ans = model.predict(test_input, BATCH_SIZE)
    AUC = round(roc_auc_score(test_label, pred_ans), 4)
    print("init----test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC : ",AUC)
              
    best_AUC = AUC
    """
    for e in range(epochs):
      for date in date_list:

        train_input = []
        train_input_pos = pd.read_pickle(
            '../model_input/din_input_'+date+'.pkl')
        train_input_neg = pd.read_pickle(
            '../model_input/din_input_'+date+'_neg.pkl')
        for i in range(len(train_input_pos)):
          feature_input = np.concatenate([train_input_pos[i],train_input_neg[i]],axis=0)
          train_input.append(feature_input)
        # model_input = np.concatenate([model_input_pos,model_input_neg],axis=1)
        train_label_pos = pd.read_pickle('../model_input/din_label_'+date+'.pkl')
        train_label_neg = pd.read_pickle('../model_input/din_label_'+date+'_neg.pkl')
        train_label = np.concatenate([train_label_pos,train_label_neg],axis=-1)

        del train_input_pos
        del train_input_neg
        del train_label_pos
        del train_label_neg
    
        n = 10
        length_train = len(train_input[0])
        length_train = int(length_train/n+1)
        
        for i in range(n):
            train_input_slice = [ column[i*length_train:(i+1)*length_train] for column in train_input]
            hist_ = model.fit(train_input_slice, train_label[i*length_train:(i+1)*length_train],
                        batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1, )
        
        # pred_ans = model.predict(test_input, BATCH_SIZE)
        #AUC = round(roc_auc_score(test_label, pred_ans), 4)
        #if AUC > best_AUC:
        #  best_AUC = AUC
        #  model.save_weights(model_dir+'/ckpt.h5')
        #print("epoch : ",e,"----date : ",date,"----test AUC : ",AUC)
