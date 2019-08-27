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
"""
model_input = pd.read_pickle(
          '../model_input/din_input_20190615.pkl')
print(model_input[12][50])
print(model_input[12][100])
print(model_input[12][200])
print(model_input[12][300])
print(model_input[12][400])
print(model_input[12][1000])
print(model_input[12][2000])
print(model_input[12][10000])
"""

with open("result_0730","r") as f:
  lines = f.readlines()

for l in lines:
  l = l.replace("\n","").replace(" ","")
  l = l.split("\t")
  items = l[1]
  if len(items)<5:
    print(l)
  items = items.split(",")
  if len(items)==0:
    print(l)
