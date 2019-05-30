# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.training import moving_averages
import os
import time
import scnutils.reader as reader
import scnutils.douban_evaluation as douban_evaluation

MOVING_AVERAGE_DECAY = 0.997
BN_EPSILON = 0.001
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)
conf = {
    "data_path": "../../data/douban/data.pkl",
    "save_path": "Gcnn_v3_test/version_2/",
    "output_path":"Gcnn_v3_output/version_2/",
    "init_model": "Gcnn_v3_model/version_2/",  # should be set for test
    "embedding_file": "../../data/douban/word_embedding.pkl",
    "CPU":"/cpu:0", #'/gpu:1'
    "emb_train":False,
    "word_embedding_dim":200,
    "batch_size": 64,  # 200 for test
    "epoch":8,
    "max_turn_num": 10,
    "max_turn_len": 50,

    "hidden_embedding_dim":200,
    "filter_size":2,
    "filter_h":3,
    "word_layers_enc":2,
    "word_layers_agg":2,
    "word_layers_itg":2,
    "_EOS_": 1,  # 1 for douban data
    "final_n_class": 1,
    "lr":0.001
}
if not os.path.exists(conf['save_path']):
    os.makedirs(conf['save_path'])
if not os.path.exists(conf['output_path']):
    os.makedirs(conf['output_path'])
if not os.path.exists(conf['init_model']):
    os.makedirs(conf['init_model'])
def bn(x, is_training, use_bias=False):
    # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    x_shape = x.get_shape()  # x_shape=[64 50 900]
    params_shape = x_shape[-1:]  # x_shape[-1:] :array([900])  x_shape[-1]:900  x_shape[:-1]:array([64 50])

    if use_bias:  # use_bias = False
        bias = tf.get_variable('bias', x_shape[-1],
                               initializer=tf.contrib.layers.xavier_initializer())  # 900
        return tf.nn.bias_add(x, bias)  # [64 50 900] 该函数要求bias是1维的，bias的维度必须和x的最后一维一样

    axis = list(range(len(x_shape) - 1))  # list(range(2))  [0 1]

    beta = tf.get_variable('beta',
                           params_shape,
                           initializer=tf.zeros_initializer())  # array([900])

    gamma = tf.get_variable('gamma',
                            params_shape,
                            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002))  # array([900])

    moving_mean = tf.get_variable('moving_mean',
                                  params_shape,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False)  # array([900])

    moving_variance = tf.get_variable('moving_variance',
                                      params_shape,
                                      initializer=tf.ones_initializer(),
                                      trainable=False)  # array([900])

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)  # mean:[4],variance [4]求向量x的均值和方差
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, MOVING_AVERAGE_DECAY, zero_debias=False)  # if zero_debias=True, has bias
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, MOVING_AVERAGE_DECAY, zero_debias=False)  #

    def mean_var_with_update():
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    if is_training:  # is_training=False
        mean, var = mean_var_with_update()
        bn_x = tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)
    else:
        bn_x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

    return bn_x  # [64 50 900]
def length(x):
    """
    :param x: tensor [64 50]
    :return:  mask_prem  (64, 50, 1)
    """
    mask_prem = tf.cast(tf.cast(tf.expand_dims(x, -1), tf.bool), tf.float32)  # type=float32
    return mask_prem
def masked_attention_axis2(x,mask):
    '''
    :param x: [64 50 50]
    :param mask:[64 1 50]
    :return:[64 50 50]
    '''
    alph=tf.multiply(x,mask) #[64 50 50] * [64 1 50]
    alph_sum=tf.reduce_sum(alph,axis=2)#
    output=tf.divide(alph,tf.expand_dims(alph_sum+(1e-10),axis=-1))
    return output

def masked_attention_axis1(x,mask):
    '''
    :param x: [64 50 50]
    :param mask:[64 50 1]
    :return:
    '''
    beta = tf.multiply(x, mask)  # [64 50 50] * (64 50 1)  #下面是0
    beta_sum = tf.reduce_sum(beta, axis=1)#(64 50)
    output = tf.divide(beta, tf.expand_dims(beta_sum+(1e-10), axis=1))#(64 50 1)

    return output

class MyModel(object):
    def __init__(self, conf):
        ## Define hyperparameters
        self.word_embedding_size = conf["word_embedding_dim"]  # 300
        emb_train = conf["emb_train"]
        self.dim = conf["hidden_embedding_dim"]  # 300
        self.max_sentence_len = conf["max_turn_len"]  # 50
        self.max_turn_num=conf["max_turn_num"]
        self.is_training = False
        self.total_words =172131   #434513
        self.rnn_units = 200
      #  self.batch_size = conf['batch_size']
        self.filter_size = conf["filter_size"]
        self.filter_h = conf["filter_h"]
        self.lr=0.001
       # self.word_embedding_size = 200

        ## Define parameters

    ## Functions

        def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                              var_scope_name="conv_layer_", reuse=None):  # padding should take attention
            '''
            :param inputs: [64 50 200]
            :param layer_idx: 0 1 2 3
            :param out_dim: 600
            :param kernel_size: 3
            :param padding:
            :param dropout:
            :param var_scope_name:
            :param reuse:
            :return:
            '''
            with tf.variable_scope(var_scope_name, reuse=reuse):
                in_dim = int(inputs.get_shape()[-1])  # 300
                V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                # V [3, 300 900]
               # inputs_look=inputs #有值
                #input_conv=tf.nn.conv1d(value=inputs, filters=V, stride=1, padding=padding) # [64 50 900] #-inf
                inputs = bn(tf.nn.conv1d(value=inputs, filters=V, stride=1, padding=padding),self.is_training)  # [64 50 600]
              #  print('inputs',inputs) #
                return inputs

        def gated_linear_units(inputs, res_inputs, last_cell, layer_idx):
            '''
            :param inputs: [64 50 900]
            :param res_inputs:  [64 50 300]
            :param last_cell: [64 50 300]
            :param layer_idx: 0
            :return:
            '''
            input_shape = inputs.get_shape().as_list()
            assert len(input_shape) == 3
            dim = int(input_shape[2])  # dim=900

            # input_gate = inputs[:,:,0:dim/4]
            forget_gate = inputs[:, :, 0:dim / 3]  # (64, 50, 300)
            output_gate = inputs[:, :, dim / 3:dim * 2 / 3]  # (64, 50, 300)
            candidate = inputs[:, :, dim * 2 / 3:]  # (64, 50, 300)

            # input_gate = tf.sigmoid(input_gate)
            forget_gate = tf.sigmoid(forget_gate)
            output_gate = tf.sigmoid(output_gate)
            candidate = tf.nn.tanh(candidate)

            if layer_idx == 0:
                cell = tf.multiply(1 - forget_gate, res_inputs)  # (64, 50, 300)
            else:
                cell = tf.multiply(forget_gate, last_cell) + tf.multiply(1 - forget_gate, res_inputs)  ##(64, 50, 300)

            output = tf.multiply(output_gate, candidate) + cell
            ##tf.multiply(x,y) #x,y维度必须相等,元素对应相等
            return output, cell  # (64, 50, 300),#(64, 50, 300)

        def linear_mapping(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping", reuse=None):
            '''
            :input 当[64 50 1200]
            :param out_dim:  300
            :param in_dim:
            :param dropout:
            :param var_scope_name:
            :param reuse:
            :return:[64 50 300]
            '''

            with tf.variable_scope(var_scope_name, reuse=reuse):
                input_shape = inputs.get_shape().as_list()  # static shape. may has None [64 50 1200]
                return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=out_dim, activation_fn=None,
                                                         weights_initializer=tf.random_normal_initializer
                                                         (mean=0,stddev=tf.sqrt(dropout * 1.0 /input_shape[-1])),
                                                         biases_initializer=tf.zeros_initializer())
            # 全连接成层

        def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, var_scope_name, reuse=None):
            '''
            nhids_list=[300 300 300 300] 当[300 300]
            kwidths_list=[3 3 3 3]   当[3 3 ]
            '''
            next_layer = inputs  # [64 50 300] 当[64 50 1200]
            cell = inputs  # [64 50 300]  当[64 50 1200]
            for layer_idx in range(len(nhids_list)):  # layer_idx=0 1 2 3   #layer_idx=0 1
                nout = nhids_list[layer_idx]  # nout=300 #nout的含义是输出维度是300
                if layer_idx == 0:  ##layer_idx=0 ,nin=300
                    nin = inputs.get_shape().as_list()[-1]  # nin=emb_dim 当 nin=1200
                else:
                    nin = nhids_list[layer_idx - 1]  # layer_idx=1,nin=300，layer_idx=2,nin=300，layer_idx=3,nin=nhids_list[2]=300
                if nin != nout:  # 在本模型中nin=nout 此处应该是防止输入向量是200的时候的情况。当 nin=1200
                    # mapping for res add
                    res_inputs = linear_mapping(next_layer, nout, dropout=dropout_dict['src'],
                                                var_scope_name=var_scope_name + "linear_mapping_cnn_" + str(layer_idx),
                                                reuse=reuse)
                else:
                    res_inputs = next_layer  ##[64 50 300]

                next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 3,
                                               kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'],
                                               var_scope_name=var_scope_name + "conv_layer_" + str(layer_idx), reuse=reuse)
                #next_layer:NAN
                # next_layer:[64 50 900]
                next_layer, cell = gated_linear_units(next_layer, res_inputs, cell, layer_idx)
                # next_layer:(64, 50, 300),cell:(64, 50, 300)
            return next_layer  # [] #维度是多少


        # Get lengths of unpadded sentences
   # def BuildModel(self):
        ## Define the placeholders
     #   start=time.time
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_turn_num, self.max_sentence_len)) #[64 10 50]
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len)) #[64 50]
        self.y_true = tf.placeholder(tf.int32, shape=(None,)) #[64]
        self.embedding_ph = tf.placeholder(tf.float32,shape=(self.total_words, self.word_embedding_size))  # [434511,200]
        self.response_len = tf.placeholder(tf.int32, shape=(None,)) #[64]
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_turn_num))#[64 10]
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size),
                                          dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)  # [64 10 50 200]
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)  # [batch_size 50 200]
        response_embeddings_T = tf.transpose(response_embeddings, perm=[0, 2, 1])
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_turn_num, axis=1) # 10个[64 50 200]
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_turn_num, axis=1)  # 10个[64]
        all_utterance_ph = tf.unstack(self.utterance_ph,num=self.max_turn_num,axis=1)# 10个[64 50] 每个句子的索引
        #A_matrix = tf.get_variable('A_matrix_v', shape=( self.dim ,  self.dim ),
         #                          initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        #B_matrix = tf.get_variable('B_matrix_v', shape=(self.dim*4, self.dim*4),
        #                           initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # tf.contrib.layers.xavier_initializer()初始化权重
        #final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse_t = None
        matching_vectors=[]
        for utterance_embeddings, utterance_len ,utterance_ph in zip(all_utterance_embeddings, all_utterance_len, all_utterance_ph):
            #utterance_embeddings:[64 50 200],utterance_len:64 utterance_ph:[64 50]
            #prem_seq_lengths, \utterance_len
            #hyp_seq_lengths self.response_len
            mask_utter = length(utterance_ph)  # length: (batch_size) mask: (batch_size, max_seq_length, 1)
            mask_response = length(self.response_ph)  # length: (64) mask: (64, 50, 1)

       # input_prem = emb(self.premise_x, mask_prem)  # 用不到mask_prem  #input_prem ：[64 50 200] #utterance_embeddings
       # input_hyp = emb(self.hypothesis_x, mask_hyp)  #response_embeddings

            dropout_dict = {'src': 1.0, 'hid': 1.0}
        # 论文中Our model has 4 convolution layers in the context representation layer and 2 layers in the aggregation layer.
        # 所以我猜FIXED_PARAMETERS["word_layers_enc"]=4
        # FIXED_PARAMETERS["word_layers_agg"]=2
         #   self.emb_utter=utterance_embeddings
            conv1_utter = conv_encoder_stack(utterance_embeddings, nhids_list=[200] * conf["word_layers_enc"],
                                        kwidths_list=[3] * conf["word_layers_enc"], dropout_dict=dropout_dict,
                                        var_scope_name="encoder_", reuse=reuse_t)

            #self.conv=conv1_utter
            conv1_response = conv_encoder_stack(response_embeddings, nhids_list=[200] * conf["word_layers_enc"],
                                       kwidths_list=[3] * conf["word_layers_enc"], dropout_dict=dropout_dict,
                                       var_scope_name="encoder_", reuse=True)
        #   conv1_hyp:(64, 50, 300)
        ### Attention ###
            self.utter_bi = conv1_utter  # (64, 50, 200)
           # print(conv1_utter.shape) #
            self.response_bi = conv1_response
            self.scores_unnorm = tf.matmul(self.utter_bi, self.response_bi, transpose_a=False, transpose_b=True)
        # tf.matmul 矩阵相乘 第一个矩阵的列数（column）等于第二个矩阵的行数（row）[64 50 50]
            self.scores_unnorm_exp=tf.exp(self.scores_unnorm)
            self.alphas = masked_attention_axis2(self.scores_unnorm_exp,tf.transpose(mask_response, perm=[0, 2, 1]))  # (batch_size,prem_len,hyp_len)
        # self.alphas:[64 50 50]
            self.betas = masked_attention_axis1(self.scores_unnorm_exp, mask_utter)  # (batch_size,prem_len,hyp_len)
        # self.betas:[64 50 50]
            response_expand = tf.tile(tf.expand_dims(self.response_bi, 1),
                                    [1, self.max_sentence_len, 1, 1])  # (batch_size,prem_len,hyp_len,hidden_dim)
        # hypothesis_expand:[64 50 50 200]
            alphas = tf.expand_dims(self.alphas, -1)  # (batch_size,prem_len,hyp_len,1)
        # alphas:[64 50 50 1]
            utter_attns = tf.reduce_sum(tf.multiply(alphas, response_expand), 2)  # (batch_size,prem_len,hidden_dim)
        # premise_attns:[64 50 200]
            utter_expand = tf.tile(tf.expand_dims(self.utter_bi, 1),
                                 [1, self.max_sentence_len, 1, 1])  # (batch_size,hyp_len,prem_len,hidden_dim)
        # premise_expand:[64 50 50 200]
            betas = tf.expand_dims(tf.transpose(self.betas, perm=[0, 2, 1]), -1)  # (batch_size,hyp_len,prem_len,1)
        # betas:[64 50 50 1]
            response_attns = tf.reduce_sum(tf.multiply(betas, utter_expand), 2)  # (batch_size,hyp_len,hidden_dim)
        # hypothesis_attns:[64 50 200]
          #  print('alphas:', self.alphas.get_shape().as_list())  # [None, prem_len, hyp_len]
          #  print('betas:', self.betas.get_shape().as_list())  # [None, prem_len, hyp_len]
          #  print('premise_attns:', utter_attns.get_shape().as_list())  # [None, prem_len, 600]
          #  print('hypothesis_attns:', response_attns.get_shape().as_list())  # [None, hyp_len, 600]
            # 这里应该是随意标注的注释
            ### Subcomponent Inference ###
            utter_diff = tf.abs(tf.subtract(self.utter_bi, utter_attns))  # [64 50 200]
            utter_mul = tf.multiply(self.utter_bi, utter_attns)
            response_diff = tf.abs(tf.subtract(self.response_bi, response_attns))
            response_mul = tf.multiply(self.response_bi, response_attns)

            m_a = tf.concat([self.utter_bi, utter_attns, utter_diff, utter_mul],2)  # premise_attns：[64 50 300] m_a：[None, prem_len, 4*200]
            m_b = tf.concat([self.response_bi, response_attns, response_diff, response_mul], 2)  # 各种维度整合的方式 #[64 50 800]
            infer_utter = conv_encoder_stack(m_a, nhids_list=[100] * conf["word_layers_agg"],
                                             kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                             var_scope_name="inference_", reuse=reuse_t)
            infer_reaponse = conv_encoder_stack(m_b, nhids_list=[100] * conf["word_layers_agg"],
                                                kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                                var_scope_name="inference_", reuse=True)
            v1_bi = infer_utter * mask_utter  # mask: (64, 50, 1) #padding 的地方归零v1_bi [64 50 300]
            v2_bi = infer_reaponse * mask_response

            v_1_sum = tf.reduce_sum(v1_bi, 1)  # v1_bi [batch_size prem_len hidden_dim] v_1_sum=[batch_size hidden_dim]
            v_1_ave = tf.div(v_1_sum,tf.expand_dims(tf.cast(utterance_len, tf.float32)+(1e-10), -1))  # [batch_size  hidden_dim]

            v_2_sum = tf.reduce_sum(v2_bi, 1)
            v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(self.response_len, tf.float32)+(1e-10),-1))  # [batch_size prem_len hidden_dim]

            v_1_max = tf.reduce_max(v1_bi, 1)
            v_2_max = tf.reduce_max(v2_bi, 1)

            v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)  # v_1_ave：[batch_size  hidden_dim]
            matching_vectors.append(v)

            if not reuse_t:
                reuse_t = True
        self.matching_vectors=matching_vectors
        self.b = tf.stack(matching_vectors, axis=1, name='matching_stack') #[64 10 800]
        self.conv2_itg = conv_encoder_stack(self.b, nhids_list=[50] * conf["word_layers_itg"],
                                         kwidths_list=[3] * conf["word_layers_itg"], dropout_dict=dropout_dict,
                                         var_scope_name="integration_", reuse=None)
        #[64 10 200]
        # = self.conv2_itg.get_shape().as_list()
        conv_2=tf.contrib.layers.flatten(self.conv2_itg)#tf.reshape(self.conv2_itg,shape=[conf["batch_size"],2000]) #[64,10*200]
        conv_2=tf.nn.dropout(conv_2,keep_prob=self.keep_prob)
        self.logits = tf.contrib.layers.fully_connected(inputs=conv_2, num_outputs=2, activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=tf.zeros_initializer())
        #self.logits = tf.layers.dense(conv2,2,activation=tf.nn.relu,name='final_v')
        self.y_pred = tf.nn.softmax(self.logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.logits ))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.total_loss)

    def Evaluate(self,sess,val_batches,score_file_path):
        labels = []
        self.all_candidate_scores = []
        val_batch_num = len(val_batches["response"])

     #   eva_score_file = open(score_file_path, 'w')
        st=0
        for batch_index in xrange(val_batch_num):
            st=st+1
            feed_dict = {self.utterance_ph: np.array(val_batches["turns"][batch_index]),
                        self.all_utterance_len_ph: np.array(val_batches["every_turn_len"][batch_index]),
                        self.response_ph: np.array(val_batches["response"][batch_index]),
                        self.response_len:np.array(val_batches["response_len"][batch_index]),
                        self.y_true: np.array(val_batches["label"][batch_index]),
                        self.keep_prob: 1
                         }
            if st%100==0:
                val_loss = sess.run(self.total_loss, feed_dict=feed_dict)
                print('val_loss',val_loss)
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])

            labels .extend(val_batches["label"][batch_index])
         #   for i in xrange(len(val_batches["label"][batch_index])):
             #  eva_score_file.write(str(candidate_scores[i]) +'\t'+str(val_batches["label"][batch_index][i])+ '\n')

      #  eva_score_file.close()
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        douban_evaluation.evaluate(all_candidate_scores,labels)
    def TrainModel(self,conf,countinue_train = False, previous_modelpath = "model"):
        start=time.time()
        conf['keep_prob']=0.7
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        print('starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        val_batches = reader.build_batches(test_data, conf)
        batch_num = len(train_data['y']) / conf["batch_size"]  # batch_num=12 500 15 625(64的时候)
        #  val_batch_num = len(val_batches["response"])
        print('batch_num', batch_num)
        conf["train_steps"] = conf["epoch"] * batch_num  # train_steps=2*3906
        conf["evaluate_step"] = max(1, batch_num / 1)  # max(1,1250) #每隔2500个batch保存一下
        conf["print_step"] = max(1, batch_num / 10)  # 1250    每隔100个batch打印一下
        print('configurations', conf)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(conf["output_path"], sess.graph)
            train_writer = tf.summary.FileWriter(conf["output_path"], sess.graph)

            with open(conf["embedding_file"], 'rb') as f:
                embeddings = pickle.load(f)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess, previous_modelpath)

            step = 0
            learning_rate = conf['lr']
            for step_i in xrange(conf["epoch"]):
                print('starting shuffle train data')
                shuffle_train = reader.unison_shuffle(train_data)  # 打乱
                train_batches = reader.build_batches(shuffle_train, conf)
                print('finish building train data')
                if step_i>1 :
                    learning_rate=learning_rate
                if step_i==2:
                    learning_rate=learning_rate*0.75
                if step_i>3 and step_i%2==0:
                    learning_rate=learning_rate*0.5
                for batch_index in range(batch_num):
                    feed_dict = {self.utterance_ph: np.array(train_batches["turns"][batch_index]),
                                 self.all_utterance_len_ph: np.array(train_batches["every_turn_len"][batch_index]),
                                 self.response_ph: np.array(train_batches["response"][batch_index]),
                                 self.response_len: np.array(train_batches["response_len"][batch_index]),
                                 self.y_true: np.array(train_batches["label"][batch_index]),
                                 self.keep_prob:conf['keep_prob'],
                                 self.lr: learning_rate
                                 }

                    _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                    train_writer.add_summary(summary)

                    step += 1
                    if step % conf["print_step"] == 0 and step > 0:  # print_step=125 一个epoch打印100次
                        print('epoch={i}'.format(i=step_i), 'step:', step, "loss",
                              sess.run(self.total_loss, feed_dict=feed_dict),
                              "processed: [" + str(step * 1.0 / batch_num) + "]")

                    if step % conf["evaluate_step"] == 0 and step > 0:  # 12500的倍数就会打印
                        index = step / conf['evaluate_step']  # evaluate_file=1250
                        score_file_path = conf['save_path'] + 'score.' + str(index)
                        self.Evaluate(sess, val_batches, score_file_path)
                        print('save evaluate_step: %s' % index)
                        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print('learning rate:',learning_rate)
                if (step_i + 1) >1:  # 模型保存6 8 10
                    saver.save(sess, os.path.join(conf["init_model"], "model.{0}".format(step_i + 1)))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i} save model'.format(i=step_i+1))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        end = time.time()
        gap = (end - start) / 3600
        print('train time:%.4f h' % gap)
    def TestModel(self, conf):
        start=time.time()
        conf['keep_prob'] = 1
        if not os.path.exists(conf['save_path']):
            os.makedirs(conf['save_path'])
        print('beging test starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')

        test_batches = reader.build_batches(test_data, conf)

        print("finish building test batches")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        # refine conf
        test_batch_num = len(test_batches["response"])

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # with tf.Session() as sess:
            # sess.run(init)
            model_name='model.6'
            saver.restore(sess, os.path.join(conf["init_model"],model_name))
            print("sucess init %s %s" % (conf["init_model"],model_name))

            score_file_path = conf['save_path'] + 'score.test'
            score_file = open(score_file_path, 'w')
            all_candidate_score = []
            labels = []
            for batch_index in xrange(test_batch_num):
                # print('utterance_ph',np.array(test_batches["turns"][batch_index]).shape)
                feed_dict = {
                    self.utterance_ph: np.array(test_batches["turns"][batch_index]),
                    # _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                    self.all_utterance_len_ph: np.array(test_batches["every_turn_len"][batch_index]),
                    self.response_ph: np.array(test_batches["response"][batch_index]),
                    self.response_len: np.array(test_batches["response_len"][batch_index]),
                    # _model.label: test_batches["label"][batch_index]
                    self.keep_prob: 1
                }
              #  last_hidden = sess.run(self.last_hidden, feed_dict=feed_dict)
               # print('last_hidden', last_hidden.shape)
                candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
                all_candidate_score.append(candidate_scores[:, 1])
                # scores = sess.run(_model.logits, feed_dict=feed)

                for i in xrange(conf["batch_size"]):
                    score_file.write(
                        str(candidate_scores[i]) + '\t' +
                        str(test_batches["label"][batch_index][i]) + '\n')
                    labels.append(test_batches["label"][batch_index][i])
            score_file.close()

            all_candidate_scores = np.concatenate(all_candidate_score, axis=0)
           # Evaluate.ComputeR10_1(all_candidate_scores, labels)
           # Evaluate.ComputeR10_2(all_candidate_scores, labels)
           # Evaluate.ComputeR10_5(all_candidate_scores, labels)
            douban_evaluation.evaluate(all_candidate_scores, labels)
            #Evaluate.ComputeR2_1(all_candidate_scores, labels)
        end = time.time()
        gap = (end - start) / 3600
        print('test time:%.4f h' % gap)
if __name__ == "__main__":
    Gcnn = MyModel(conf)
    Gcnn.TrainModel(conf)
  #  Gcnn.TestModel(conf)
    '''
    ### Inference Composition ###
    # infer_prem=[batch_size prem_len hidden_dim] ?  [batch_size prem_len 2*hidden_dim]
    infer_utter = conv_encoder_stack(m_a, nhids_list=[200] * conf["word_layers_agg"],
                                    kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                    var_scope_name="inference_", reuse=None)
    infer_reaponse = conv_encoder_stack(m_b, nhids_list=[200] * conf["word_layers_agg"],
                                   kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                   var_scope_name="inference_", reuse=True)
    # infer_prem:[64 50 300]
    ### Pooling Layer ###
    v1_bi = infer_utter * mask_utter  # mask: (64, 50, 1) #padding 的地方归零v1_bi [64 50 300]
    v2_bi = infer_reaponse * mask_response

    v_1_sum = tf.reduce_sum(v1_bi, 1)  # v1_bi [batch_size prem_len hidden_dim] v_1_sum=[batch_size hidden_dim]
    v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(utterance_len, tf.float32), -1))  # [batch_size  hidden_dim]

    v_2_sum = tf.reduce_sum(v2_bi, 1)
    v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(self.response_len, tf.float32), -1))  # [batch_size prem_len hidden_dim]

    v_1_max = tf.reduce_max(v1_bi, 1)
    v_2_max = tf.reduce_max(v2_bi, 1)

    v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)  # v_1_ave：[batch_size  hidden_dim]
    # v [batch_size  hidden_dim*4]

    # MLP layer
    v_shape = v.get_shape().as_list()
    self.W_mlp = tf.Variable(tf.random_normal([v_shape[-1], self.dim], stddev=0.1))
    self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

    self.W_cl = tf.Variable(tf.random_normal([self.dim, 2], stddev=0.1))
    self.b_cl = tf.Variable(tf.random_normal([2], stddev=0.1))

    h_mlp = tf.nn.relu(tf.matmul(v, self.W_mlp) + self.b_mlp)

    # Dropout applied to classifier
    h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

    # Get prediction
    self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

    # Define the cost function
    self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
    print('self.total_cost:', self.total_cost.get_shape().as_list())
    for ele in tf.global_variables():
        print(ele.op.name)
    '''
