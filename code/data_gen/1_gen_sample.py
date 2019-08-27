# coding: utf-8
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import gc
from joblib import Parallel, delayed

def gen_session_list_din(uid, t):
  session_list = [[] for i in range(len(date_list))]
  
  for row in t.iterrows():
    date = row[1]['date']
    i = date_list.index(str(date))
    item_id = row[1]['item_id']
    session_list[i].append(item_id)

  return uid, session_list

def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}

if __name__ == "__main__":
  date_list = ["20190610","20190611","20190612","20190613","20190614","20190615","20190616","20190617","20190618","20190619","20190620",]

  if not os.path.exists('../sampled_data/'):
    os.mkdir('../sampled_data/')

  data = pd.read_csv("../raw_data/ECommAI_ubp_round1_train",sep="\t",header=None, names=["user_id","item_id","behavior","date"])
  train_users_items = {
    "user_id":np.unique(data["user_id"]),
    "item_id":np.unique(data["item_id"])
  }
  id_record = {}
  test_users = pd.read_csv('../raw_data/ECommAI_ubp_round1_test',sep="\t",header=None, names=["user_id"])
  
  user_feature_name = ["user_id","pred_gender","pred_age_level","pred_education_degree","pred_career_type","predict_income","pred_stage"]
  users = pd.read_csv("../raw_data/ECommAI_ubp_round1_user_feature", sep="\t", header=None, names=user_feature_name)
  
  users = users.fillna(-1)

  user_id_unique = np.unique(np.concatenate([train_users_items["user_id"],test_users["user_id"],users["user_id"]]))
  id_record["user"] = user_id_unique

  lbe = LabelEncoder()
  lbe.fit(user_id_unique)
  users["user_id"] = lbe.transform(users["user_id"])
  data["user_id"] = lbe.transform(data["user_id"])

  lbe = LabelEncoder()
  users["pred_gender"] = lbe.fit_transform(users["pred_gender"])
  
  lbe = LabelEncoder()
  users["pred_age_level"] = lbe.fit_transform(users["pred_age_level"])

  lbe = LabelEncoder()
  users["pred_education_degree"] = lbe.fit_transform(users["pred_education_degree"])

  lbe = LabelEncoder()
  users["pred_career_type"] = lbe.fit_transform(users["pred_career_type"])

  lbe = MinMaxScaler(feature_range=(0,1))
  users["predict_income"] = lbe.fit_transform(users[["predict_income"]])

  lbe = LabelEncoder()
  pred_stage = []
  for i in range(len(users["pred_stage"])):
    if isinstance(users["pred_stage"][i],float) and math.isnan(users["pred_stage"][i]):
      tmp = []
    else:
      tmp = str(users["pred_stage"][i]).split(",")
    pred_stage.append(tmp)
  pred_stage_uniq = np.unique(np.concatenate(pred_stage))
  print(pred_stage_uniq)
  lbe.fit(pred_stage_uniq)
  for i in range(len(pred_stage)):
    pred_stage[i] = lbe.transform(pred_stage[i])+1
  users["pred_stage"] = pred_stage
  
  print("user example ----------------")
  print(users[:3])
  pd.to_pickle(users, '../sampled_data/users_feature.pkl')
  del users
  print("user feature ok")

  item_feature_name = ["item_id","cate_1_id","cate_id","brand_id","price"]
  items = pd.read_csv("../raw_data/ECommAI_ubp_round1_item_feature", sep="\t", header=None, names=item_feature_name)
  
  items = items.fillna(-1)

  item_id_unique = np.unique(np.concatenate([train_users_items["item_id"], items["item_id"]]))
  id_record["item"] = item_id_unique
  pd.to_pickle(id_record, '../sampled_data/id_record.pkl')

  lbe = LabelEncoder()
  lbe.fit(item_id_unique)
  items["item_id"] = lbe.transform(items["item_id"])+1
  data["item_id"] = lbe.transform(data["item_id"])+1

  lbe = LabelEncoder()
  items["cate_1_id"] = lbe.fit_transform(items["cate_1_id"])
  
  lbe = LabelEncoder()
  items["cate_id"] = lbe.fit_transform(items["cate_id"])
  
  lbe = LabelEncoder()
  items["brand_id"] = lbe.fit_transform(items["brand_id"])

  lbe = MinMaxScaler(feature_range=(0,1))
  items[["price"]] = lbe.fit_transform(items[["price"]])

  print("item example ----------------")
  print(items[:3])

  pd.to_pickle(items, '../sampled_data/items_feature.pkl')
  del items
  print("item feature ok")

  data = data.groupby('user_id')
  user_hist_session = applyParallel(data, gen_session_list_din, n_jobs=20, backend='loky')

  print("user_hist_session example -------------")
  print(user_hist_session[user_id_unique[0]])

  pd.to_pickle(user_hist_session, '../sampled_data/user_hist_session.pkl')
  gc.collect()
  
  print("1_gen_sampled_data done")
