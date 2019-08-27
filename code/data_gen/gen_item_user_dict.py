# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc
from joblib import Parallel, delayed
import time

def gen_item_user_dict(item_id, t):
  user_list = np.unique(t["user_id"])
  # print(user_list)
  """
  for row in t.iterrows():
    user_id = row[1]['user_id']
    user_list.append(user_id)
  """
  # user_list = list(set(user_list))
  return item_id, user_list

def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}
"""
if __name__ == "__main__":
  data = pd.read_csv("../raw_data/ECommAI_ubp_round1_train",sep="\t",header=None, names=["user_id","item_id","behavior","date"])
  # data = pd.read_csv("../raw_data/train_sample",sep="\t",header=None, names=["user_id","item_id","behavior","date"])
  print(data.shape)
  data = data.groupby('item_id')
  item_user_dict = applyParallel(data, gen_item_user_dict, n_jobs=20, backend='loky')
  if not os.path.exists('../sampled_data/'):
    os.mkdir('../sampled_data/')
  pd.to_pickle(item_user_dict, '../sampled_data/item_user_dict_raw_id.pkl')
  gc.collect()
"""
n = 0
item_user_dict = {}
start = time.time()
with open("../raw_data/ECommAI_ubp_round1_train","r") as f:
  line = f.readline()
  while line:
    l = line.split("\t")
    user_id = l[0]
    item_id = l[1]
    if item_id not in item_user_dict:
      item_user_dict[item_id] = [user_id]
    else:
      item_user_dict[item_id].append(user_id)
    line = f.readline()
    n+=1
    if n%100000==0:
      end = time.time()
      print(n,"----",int(end-start))
      start = time.time()
for item_id in item_user_dict.keys():
  item_user_dict[item_id] = list(set(item_user_dict[item_id]))
pd.to_pickle(item_user_dict, '../sampled_data/item_user_dict_raw_id2.pkl')





