# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc
from joblib import Parallel, delayed
import time,datetime
import pickle

date_list = ["20190810","20190811","20190812","20190813","20190814","20190815","20190816","20190817","20190818","20190819","20190820",]

def show_time():
  return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# def gen_session_list_din(uid, t):
#   session_list = [[] for i in range(len(date_list))]
  
#   for row in t.iterrows():
#     date = row[1]['date']
#     i = date_list.index(str(date))
#     item_id = row[1]['item_id']
#     session_list[i].append(item_id)

#   return uid, session_list

# def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
#     """Use Parallel and delayed """  # backend='threading'
#     results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
#         delayed(func)(name, group) for name, group in df_grouped)

#     return {k: v for k, v in results}



# if __name__ == "__main__":
#   data = pd.read_csv("../raw_data/ECommAI_ubp_round1_train",sep="\t",header=None, names=["user_id","item_id","behavior","date"])
#   print(data.shape)
#   data = data.groupby('user_id')
#   user_hist_session = applyParallel(data, gen_session_list_din, n_jobs=20, backend='loky')
#   if not os.path.exists('../sampled_data/'):
#     os.mkdir('../sampled_data/')
#   pd.to_pickle(user_hist_session, '../sampled_data/user_hist_session_raw_id.pkl')
#   gc.collect()

"""
if __name__ == "__main__":
  print("start program: ",show_time())
  user_behavior_dict = {}
  read_num = 0
  start = time.time()
  with open("../download/ECommAI_ubp_round2_train","r") as f:
    behavior = f.readline()
    while behavior:
      b = behavior.strip().split("\t")
      user_id = b[0]
      item_id = b[1]
      # action = b[2]
      date = b[3]
      
      if user_id in user_behavior_dict:
        i = int(date[-2:])-10
        user_behavior_dict[user_id][i].append(item_id)
      else:
        user_behavior_dict[user_id] = [[] for i in range(len(date_list))]
        i = int(date[-2:])-10
        user_behavior_dict[user_id][i].append(item_id)
      read_num += 1
      if read_num % 10000000 == 0 :
        end = time.time()
        print(read_num,"----",int(end-start))
        start = time.time()
      behavior = f.readline()
  print("cache user behavior dict: ",show_time())

  if not os.path.exists('../sampled_data/'):
    os.mkdir('../sampled_data/')
  # pd.to_pickle(user_behavior_dict, '../sampled_data/user_hehavior.pkl')
  with open('../sampled_data/user_hehavior.pkl',"wb") as f:
    pickle.dump(user_behavior_dict,f)
  print("write user behavior dict: ",show_time())

  for user_id in user_behavior_dict.keys():
    user_behavior_dict[user_id] = list(np.unique(np.concatenate(user_behavior_dict[user_id])))
  print("cache user behavior unique dict: ",show_time())
  # pd.to_pickle(user_behavior_dict, '../sampled_data/user_hehavior_unique.pkl')
  with open('../sampled_data/user_hehavior_unique.pkl','wb') as f:
    pickle.dump(user_behavior_dict,f)
  print("write user behavior unique dict: ",show_time())
"""

if __name__ == "__main__":
  if not os.path.exists('../sampled_data/'):
    os.mkdir('../sampled_data/')
  
  cache_user_id = "1084105303"
  cache_item_id_list = []

  start = time.time()
  read_num = 0
  with open("../download/ECommAI_ubp_round2_train_sort","r") as f:
    with open('../sampled_data/user_hehavior_unique',"w") as wf:
      behavior = f.readline()
      while behavior:
        b = behavior.strip().split("\t")
        user_id = b[0]
        if user_id != cache_user_id:
          wf.write(user_id + "\t" + ",".join(list(np.unique(cache_item_id_list)))+"\n")
          cache_user_id = user_id
          cache_item_id_list = []
        item_id = b[1]
        # action = b[2]
        # date = b[3]
        cache_item_id_list.append(item_id)
        read_num += 1
        if read_num % 10000000 == 0 :
          end = time.time()
          print(read_num,"----",int(end-start))
          start = time.time()
      wf.write(user_id + "\t" + ",".join(list(np.unique(cache_item_id_list))))
