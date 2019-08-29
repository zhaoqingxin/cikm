import os
import numpy as np
import pandas as pd
import time,datetime
import random
import 

def show_time():
  return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

date_list = ["20190810","20190811","20190812","20190813","20190814","20190815","20190816","20190817","20190818","20190819","20190820",]

print("cache behavior and items: ", show_time())

# user_behavior = pd.read_pickle("../sampled_data/user_hehavior_unique.pkl")
with open("../sampled_data/user_hehavior_unique.pkl","rb") as f:
  user_behavior = pickle.load(f)
print("user_behavior length: ", len(user_behavior))
user_abc = user_behavior.keys()
print(user_behavior[user_abc[8765]])
print(user_behavior[user_abc[234]])
user_abc = None


with open("../sampled_data/train_items","r") as f:
  items = f.readlines()
items_length = len(items)
print("items_length: ",items_length)

sharding_num = 1000
pos_sample_num = 796996580
sample_per_file = pos_sample_num//sharding_num+1

read_num = 0
file_num = 0

def num4(num):
  s = "000" + str(num)
  return s[-4:]
start = time.time()

print("start gen neg_sample: ", show_time())
with open("../download/ECommAI_ubp_round2_train","r") as f:
  behavior = f.readline()
  pos_wf = open("../sampled_data/sharding/train_pos_"+num4(file_num),"w")
  neg_wf = open("../sampled_data/sharding/train_neg1_"+num4(file_num),"w")
  
  while behavior:
    pos_wf.write(behavior)
    b = behavior.split("\t")
    while True:
      neg_item_index = random.randint(0,items_length-1)
      neg_item = items[neg_item_index].strip()
      if neg_item not in user_behavior[b[0]]:
        b[1] = neg_item
        break
    neg_wf.write("\t".join(b))
    read_num += 1
    if read_num % sample_per_file == 0:
      neg_wf.close()
      pos_wf.close()

      file_num += 1
      pos_wf = open("../sampled_data/sharding/train_pos_"+num4(file_num),"w")
      neg_wf = open("../sampled_data/sharding/train_neg1_"+num4(file_num),"w")
      
    if read_num % 10000000 == 0 :
      end = time.time()
      print(file_num,"----",int(end-start))
      start = time.time()
    behavior = f.readline()
print("end gen neg_sample: ", show_time())


