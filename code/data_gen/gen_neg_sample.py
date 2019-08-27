import os
import numpy as np
import pandas as pd
import time
import random

date_list = ["20190610","20190611","20190612","20190613","20190614","20190615","20190616","20190617","20190618","20190619","20190620",]

user_sess = pd.read_pickle("../sampled_data/user_hist_session.pkl")
for user in user_sess.keys():
  user_sess[user] = np.unique(user_sess[user])

items = pd.read_csv("../raw_data/ECommAI_ubp_round1_item_feature",sep="\t",header=None, names=["item_id","cate_1_id","cate_id","brand_id","price"])
items = items['item_id']
length = len(items)-1

data = pd.read_csv("../raw_data/ECommAI_ubp_round1_train",sep="\t",header=None, names=["user_id","item_id","behavior","date"])

users = data["user_id"]
neg_items = []

print("neg item start gen----------------")

start = time.time()
n = 0
for user in users:
  while True:
    i = random.randint(0,length)
    item = items[i]
    if item not in user_sess[user]:
      assert n == len(neg_items)
      neg_items.append(item)
      break
  n+=1
  if n %1000000 == 0:
    end = time.time()
    print(n,"----",int(end-start))
    start = time.time()

print("neg_items length: ",len(neg_items))
data["item_id"] = neg_items

data.to_csv('../raw_data/ECommAI_ubp_round1_train_neg',sep="\t",header=False,index=False)


