# coding: utf-8

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_fixlen_feature_names,get_varlen_feature_names
import time

DIN_SESS_MAX_LEN=100

date_list = ["20190610","20190611","20190612","20190613","20190614","20190615","20190616","20190617","20190618","20190619","20190620"]

def gen_sess_feature_din(user_id,date):
    sess_max_len = DIN_SESS_MAX_LEN
    hist_items = [0]
    sess_input_length = 0
    if user_id in user_hist_session:
        idx = date_list.index(str(date))
        hist_items = user_hist_session[user_id][:idx]
        if len(hist_items)>0:
            hist_items = np.concatenate(hist_items)
        hist_items = hist_items[-sess_max_len:]
        sess_input_length = len(hist_items)
    return hist_items, sess_input_length


if __name__ == "__main__":

    user_hist_session = pd.read_pickle('../sampled_data/user_hist_session.pkl')

    users_feature = pd.read_pickle('../sampled_data/users_feature.pkl')
    items_feature = pd.read_pickle('../sampled_data/items_feature.pkl')
    id_record = pd.read_pickle('../sampled_data/id_record.pkl')
    user_id_unique = id_record["user"]
    item_id_unique = id_record["item"]

    for date in date_list[4:]:
        for f in ["","_neg"]:
            start = time.time()
            currentTimeStamp = time.time()
            time_local = time.localtime(currentTimeStamp)
            time_YmdHMS = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print(date+f,' start time : ', time_YmdHMS)
            sample_sub = pd.read_csv("../raw_data/train_"+date+f,sep="\t",header=None, names=["user_id","item_id","behavior","date"])
            
            lbe = LabelEncoder()
            lbe.fit(user_id_unique)
            sample_sub["user_id"] = lbe.transform(sample_sub["user_id"])

            lbe = LabelEncoder()
            lbe.fit(item_id_unique)
            sample_sub["item_id"] = lbe.transform(sample_sub["item_id"])+1

            sess_input_dict = {'hist_item_id':[]}
            sess_input_length = []
            sess_cache = {}
            for user_id in sample_sub['user_id']:
                if user_id in sess_cache:
                    sess_input_dict['hist_item_id'].append(sess_cache[user_id]["sess"])
                    sess_input_length.append(sess_cache[user_id]["length"])
                else:
                    hist_items, length = gen_sess_feature_din(user_id,date)
                    sess_input_dict['hist_item_id'].append(hist_items)
                    sess_input_length.append(length)
                    sess_cache[user_id] = {
                        "sess":hist_items,
                        "length":length
                    }
            currentTimeStamp = time.time()
            time_local = time.localtime(currentTimeStamp)
            time_YmdHMS = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print('sess ok, time : ', time_YmdHMS)
 
            sample_sub = pd.read_csv("../raw_data/train_"+date+f,sep="\t",header=None, names=["user_id","item_id","behavior","date"])
            lbe = LabelEncoder()
            lbe.fit(user_id_unique)
            sample_sub["user_id"] = lbe.transform(sample_sub["user_id"])

            lbe = LabelEncoder()
            lbe.fit(item_id_unique)
            sample_sub["item_id"] = lbe.transform(sample_sub["item_id"])+1

            currentTimeStamp = time.time()
            time_local = time.localtime(currentTimeStamp)
            time_YmdHMS = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print('data read ok, time : ', time_YmdHMS)

            data = pd.merge(sample_sub, users_feature, how='left', on='user_id', )
            data = pd.merge(data, items_feature, how='left', on='item_id')
            data['predict_income'] = data['predict_income'].fillna(0.)
            data['price'] = data['price'].fillna(0.)
            data = data.fillna(-1)

            user_sparse_features = ["user_id", "pred_gender","pred_age_level","pred_education_degree","pred_career_type"]
            item_sparse_features = ["item_id", "cate_1_id","cate_id","brand_id"]
            dense_features = ["predict_income",'price']
            varLen_sparse_features = ["pred_stage"]
            hist_feature = ["hist_item_id"]

            for feat in user_sparse_features+item_sparse_features:
                data[feat] = data[feat].astype(np.int32)

            for feat in user_sparse_features:
                if feat != "user_id":
                    data[feat] = data[feat]+1
            for feat in item_sparse_features:
                if feat != "item_id":
                    data[feat] = data[feat]+1
            
            feature_list = []
            for feat in user_sparse_features:
                if feat != "user_id":
                    feature_list.append(SparseFeat(feat, users_feature[feat].nunique() + 1))
                else:
                    feature_list.append(SparseFeat(feat, len(user_id_unique)))

            for feat in item_sparse_features:
                if feat != "item_id":
                    feature_list.append(SparseFeat(feat, items_feature[feat].nunique() + 1))
                else:
                    feature_list.append(SparseFeat(feat, len(item_id_unique)+1))

            dense_feature_list = [DenseFeat(feat, 1) for feat in dense_features]

            varLen_sparse_feature_list = [VarLenSparseFeat(feat,11, maxlen=10) for feat in varLen_sparse_features]

            sess_sparse_feature_list = [VarLenSparseFeat(feat, len(item_id_unique)+1, maxlen=DIN_SESS_MAX_LEN, embedding_name='item_id') for feat in hist_feature]

            feature_list = feature_list + dense_feature_list + varLen_sparse_feature_list

            # pred_stage = [ i for i in data["pred_stage"]]
            pred_stage = []
            for i in data["pred_stage"]:
                if type(i) is np.ndarray:
                    pred_stage.append(i)
                else:
                    pred_stage.append(np.array([]))
            pred_stage = pad_sequences(pred_stage, maxlen=10, padding='post')

            sess_input = [pad_sequences(sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post') for feat in hist_feature]
            #model_input = [data[feat.name].values for feat in feature_list] 
            model_input = []
            for feat in feature_list:
                if feat.name=="pred_stage":
                    model_input.append(pred_stage)
                else:
                    model_input.append(data[feat.name].values)

            model_input += sess_input

            feature_list += sess_sparse_feature_list

            label_shape = data["behavior"].shape
            if f=="_neg":
                label = np.zeros(label_shape)
            else:
                label = np.ones(label_shape)
            
            currentTimeStamp = time.time()
            time_local = time.localtime(currentTimeStamp)
            time_YmdHMS = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print('data transform ok, time : ', time_YmdHMS)

            if not os.path.exists('../model_input/'):
                os.mkdir('../model_input/')

            pd.to_pickle(model_input, '../model_input/din_input_'+date+f+'.pkl')
            # pd.to_pickle([np.array(sess_input_length)], '../model_input/din_input_len.pkl')
            pd.to_pickle(label, '../model_input/din_label_'+date+f+'.pkl')
            pd.to_pickle(feature_list, '../model_input/din_fd.pkl', )
            currentTimeStamp = time.time()
            time_local = time.localtime(currentTimeStamp)
            time_YmdHMS = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print('data write ok, time : ', time_YmdHMS)
            end = time.time()
            print(date+f,"----",int(end-start))
    print("gen din input done")
