# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc
from joblib import Parallel, delayed

date_list = ["20190610","20190611","20190612","20190613","20190614","20190615","20190616","20190617","20190618","20190619","20190620",]

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
  data = pd.read_csv("../raw_data/ECommAI_ubp_round1_train",sep="\t",header=None, names=["user_id","item_id","behavior","date"])
  print(data.shape)
  data = data.groupby('user_id')
  user_hist_session = applyParallel(data, gen_session_list_din, n_jobs=20, backend='loky')
  if not os.path.exists('../sampled_data/'):
    os.mkdir('../sampled_data/')
  pd.to_pickle(user_hist_session, '../sampled_data/user_hist_session_raw_id.pkl')
  gc.collect()
