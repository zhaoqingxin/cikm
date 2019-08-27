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

user_hist_session = pd.read_pickle('../sampled_data/user_hist_session_raw_id.pkl')

test_users = pd.read_csv('../raw_data/ECommAI_ubp_round1_test',sep="\t",header=None, names=["user_id"])

# item_top_50 = "1179543312,331431843,883229628,349064152,325284569,1238146133,1213502497,862784080,856488129,1073855408,648337049,325413168,527376219,1027136273,1034988387,882946649,983522798,208789868,135247206,1240479781,1215797515,1095080482,528409930,11098051,693634720,691870025,966924067,1058970420,1132202973,877239449,1187675444,917992086,1207463578,701222471,971801790,856940750,510961059,271197270,1229789483,1120477236,1101105159,699083056,518881107,696345539,1181415108,526120217,1247563020,896127701,865472887,854585570"
# tem_top_50 = item_top_50.split(",")
with open("interaction_recall_200items","w") as f:
  for u in test_users["user_id"]:
    if u not in user_hist_session:
      # f.write(str(u)+"\t"+",".join(tem_top_50)+"\n")
      f.write(str(u)+"\n")
      continue
    hist = np.concatenate(user_hist_session[u])
    hist = np.unique(hist,return_counts=True)
    if len(hist[0]) < 200:
      if len(hist[0])==0:
        f.write(str(u)+"\n")
      else:
        recommend = [str(int(i)) for i in hist[0]]
        recommend = ",".join(recommend)
        f.write(str(u)+"\t"+recommend+"\n")
      continue
    hist = list(zip(hist[1],hist[0]))
    hist.sort(reverse=True)
    recommend = [ str(int(j)) for i,j in hist ]
    recommend = ",".join(recommend[:200])
    f.write(str(u)+"\t"+recommend+"\n")



