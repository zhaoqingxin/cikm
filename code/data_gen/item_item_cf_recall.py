# coding: utf-8

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
import _thread
from multiprocessing import Process

user_hist_session = pd.read_pickle('../sampled_data/user_hist_session_raw_id.pkl')
for user in user_hist_session.keys():
  # print(user,"----",type(user))
  # print(user_hist_session[user],"----",type(user_hist_session[user][0][0]))
  user_hist_session[user]=np.unique(np.concatenate(user_hist_session[user]))
  user_hist_session[user] = [str(int(item)) for item in user_hist_session[user]]
  # break
print("read session ok!")
item_user_dict = pd.read_pickle('../sampled_data/item_user_dict_raw_id2.pkl')
print("read item_user dict ok!")

def fn(user):
  items = []
  if user in user_hist_session:
    items = user_hist_session[user]
  similar_users = []
  for item in items:
    if item in item_user_dict:
      similar_users.append(item_user_dict[item])
  """
  if len(similar_users)>0:
    similar_users = np.unique(np.concatenate(similar_users))
  if len(similar_users)>50:
    similar_users = np.random.choice(similar_users,50)
  """
  if len(similar_users)>0:
    similar_users = np.unique(np.concatenate(similar_users),return_counts=True)
    if len(similar_users[0])<=50:
      similar_users = similar_users[0]
    else:
      similar_users = list(zip(similar_users[1],similar_users[0]))
      similar_users.sort(reverse=True)
      similar_users = [ j for i,j in similar_users[:50]]
  
  recall = []
  for u in similar_users:
    if int(u) in user_hist_session:
      recall.append(user_hist_session[int(u)])
  if len(recall) > 0:
    recall = np.unique(np.concatenate(recall),return_counts=True)
    if len(recall[0]) > 0:
      recall = list(zip(recall[1],recall[0]))
      recall.sort(reverse=True)
      recall = [ j for i,j in recall ]
    else:
      recall = []
  recall = list(filter(lambda x: x not in items, recall))
  recall = recall[:200]
  if len(recall) == 0:
    return str(user)+"\n"
  else:
    return str(user)+"\t"+",".join(recall)+"\n"

test_users = pd.read_csv('../raw_data/ECommAI_ubp_round1_test',sep="\t",header=None, names=["user_id"])
print("start-----------------------compute")

def tod(i):
  i = str(i)
  if len(i)==1:
    i = "0"+i
  return i

def fnfn(threadName,user_ids):
  print(len(user_ids))
  n = 0
  start = time.time()
  with open("0806/"+threadName,"w") as f:
    for user_id in user_ids:
      line = fn(user_id)
      f.write(line)
      n += 1
      if n%1000==0:
        end = time.time()
        print(n,"----",int(end-start))
        start = time.time()

if __name__ == "__main__":
  # result = Parallel(n_jobs=20,verbose=4)(delayed(fn)(user_id) for user_id in test_users["user_id"])
  length = len(test_users["user_id"])//20+1
  ps = []
  for i in range(20):
    ps.append(Process( target=fnfn, args=("p-"+tod(i), test_users["user_id"][i*length:(i+1)*length])))
  for p in ps:
    p.start()
    p.join()







