# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""


from tensorflow.python.keras.layers import Dense,Concatenate, Flatten,Embedding
from tensorflow.python.keras.models import Model

from ..inputs import  build_input_features,SparseFeat,VarLenSparseFeat,DenseFeat,get_dense_input,varlen_embedding_lookup,get_varlen_pooling_list,combined_dnn_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.utils import concat_fun, NoMask


def create_embedding_matrix(sparse_feature_columns):
    # return { sparse_feature_column.embedding_name : Embedding(sparse_feature_column.dimension, 2 * int(pow(sparse_feature_column.dimension, 0.25)),name='emb_' + sparse_feature_column.embedding_name)
    return { sparse_feature_column.embedding_name : Embedding(sparse_feature_column.dimension, 4)
        for sparse_feature_column in sparse_feature_columns }

def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.embedding:
            lookup_idx = sparse_input_dict[feature_name]
            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return embedding_vec_list

def FC(dnn_feature_columns, history_feature_list, embedding_size=8, hist_len_max=16, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', 
       l2_reg_dnn=0,  dnn_dropout=0, init_std=0.0001, seed=1024,
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

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(sparse_feature_columns)



    dnn_input_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    print(dense_value_list)
    deep_input_emb = concat_fun(dnn_input_emb_list)


    deep_input_emb = deep_input_emb
    print(deep_input_emb)
    deep_input_emb = Flatten()(deep_input_emb)
    print(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb],dense_value_list)
    print(dnn_input)
    # output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
    #              dnn_dropout, dnn_use_bn, seed)(dnn_input)

    output = Dense(200, activation='relu')(dnn_input)
    output = Dense(80, activation='relu')(output)

    output = Dense(1, activation='sigmoid')(output)

    # output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model

