# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""


from tensorflow.python.keras.layers import Dense,Concatenate, Flatten
from tensorflow.python.keras.models import Model

from ..inputs import  build_input_features,create_embedding_matrix,SparseFeat,VarLenSparseFeat,DenseFeat,embedding_lookup,get_dense_input,varlen_embedding_lookup,get_varlen_pooling_list,combined_dnn_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.utils import concat_fun, NoMask


def DIN(dnn_feature_columns, history_feature_list, embedding_size=8, hist_len_max=16, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """


    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    # varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []


    inputs_list = list(features.values())


    embedding_dict = create_embedding_matrix(dnn_feature_columns,l2_reg_embedding,init_std,seed,embedding_size, prefix="")


    dnn_input_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns,mask_feat_list=history_feature_list)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    deep_input_emb = concat_fun(dnn_input_emb_list)


    deep_input_emb = NoMask()(deep_input_emb)
    print(deep_input_emb)
    deep_input_emb = Flatten()(deep_input_emb)
    print(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb],dense_value_list)
    print(dnn_input)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(dnn_input)
    final_logit = Dense(1, use_bias=False)(output)

    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
