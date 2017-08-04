import tensorflow as tf
import sys

from tensorflow.contrib.rnn import BasicLSTMCell
from BiDAF_ops.my.tensorflow.nn import softsel, get_logits, highway_network
from BiDAF_ops.my.tensorflow.rnn import bidirectional_dynamic_rnn
from BiDAF_ops.my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from model import Basic
from ops import *


class BiDAF(Basic):
    def __init__(self, params, initializer):
        self.input_keep_prob = params['input_keep_prob']
        self.share_lstm_weights = params['share_lstm_weights']
        self.wd = params['wd']
        self.logit_func = params['logit_func']
        self.answer_func = params['answer_func']
        self.highway = params['highway']
        self.highway_num_layers = params['highway_num_layers']
        self.hidden_size = params['hidden_size']
        self.load_seo = params['load_seo']
        # Placeholders
        self.is_train = tf.placeholder('bool')
        
        super(BiDAF, self).__init__(params, initializer)


    def init_bidaf(self):
        self.cell_fw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.cell_bw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.d_cell_fw = SwitchableDropoutWrapper(self.cell_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell_bw = SwitchableDropoutWrapper(self.cell_bw, self.is_train,\
                    input_keep_prob=self.input_keep_prob)
        self.cell2_fw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.cell2_bw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.d_cell2_fw = SwitchableDropoutWrapper(self.cell2_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell2_bw = SwitchableDropoutWrapper(self.cell2_bw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.cell3_fw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.cell3_bw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.d_cell3_fw = SwitchableDropoutWrapper(self.cell3_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell3_bw = SwitchableDropoutWrapper(self.cell3_bw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.cell4_fw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.cell4_bw = BasicLSTMCell(self.dim_rnn_cell, state_is_tuple=True)
        self.d_cell4_fw = SwitchableDropoutWrapper(self.cell4_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell4_bw = SwitchableDropoutWrapper(self.cell4_bw, self.is_train, \
                  input_keep_prob=self.input_keep_prob)

        self.x_mask = tf.sequence_mask(lengths=self.context_len, maxlen=self.context_maxlen)
        self.x_mask = tf.expand_dims(self.x_mask, 1)
        self.q_mask = tf.sequence_mask(lengths=self.question_len, maxlen=self.question_maxlen)
       
        context_embed = dropout(embedding_lookup(
                inputs=self.context,
                voca_size=self.voca_size,
                embedding_dim=self.dim_embed_word, 
                initializer=self.initializer, 
                trainable=self.embed_trainable,
                reuse=True, scope='Word'), self.embed_dropout)
        
        question_embed = dropout(embedding_lookup(
                inputs=self.question,
                voca_size=self.voca_size,
                embedding_dim=self.dim_embed_word,
                initializer=self.initializer,
                trainable=self.embed_trainable,
                reuse=True, scope='Word'), self.embed_dropout)
        
        char_context_embed, char_question_embed = self.character_embedding_layer(
                self.context_char, self.question_char,
                self.char_size, self.char_emb_dim,
                self.char_out, self.filter_width,
                self.cnn_keep_prob, self.share_conv)

        context_embed_input = dropout(tf.concat(
                [context_embed, char_context_embed],2), self.embed_dropout)
        question_embed_input = dropout(tf.concat(
                [question_embed, char_question_embed],2),self.embed_dropout)

        self.N = context_embed_input.get_shape().as_list()[0]
        self.JX = context_embed_input.get_shape().as_list()[1]
        self.JQ = question_embed_input.get_shape().as_list()[1]
        self.con_len = tf.expand_dims(self.context_len, 1)
        xx = tf.expand_dims(context_embed_input, 1)
        qq = question_embed_input
        if self.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, self.highway_num_layers, True, wd=self.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, self.highway_num_layers, True, wd=self.wd, is_train=self.is_train)
        return xx, qq
    

    def character_embedding_layer(self, context_char, question_char, char_size, 
            char_emb_dim, char_out, filter_width, cnn_keep_prob, share_conv):
        
        def char_conv(inputs, emb_dim, output_dim, 
                filter_width, padding,keep_prob = 1.0,scope = None):
            with tf.variable_scope('conv' or scope):
                num_channels = emb_dim
                with tf.variable_scope('xx/conv1d_5'):
                    conv_filter = tf.get_variable("filter", 
                            shape=[1,filter_width, num_channels, output_dim], 
                            dtype=tf.float32)
                    bias = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32)
                strides = [1,1,1,1]
                inputs = dropout(inputs, keep_prob)
                conv = tf.nn.conv2d(inputs, conv_filter, strides, padding) + bias
                conv_output = tf.reduce_max(tf.nn.relu(conv),2)
                return conv_output
        
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"):
                char_emb_matrix = tf.get_variable(
                        "char_emb_mat", shape = [(char_size),char_emb_dim],
                        dtype = tf.float32, trainable = True)
                   # char_emb_pad = tf.constant(([[0.0]*self.char_emb_dim]),
                   #    dtype = tf.float32)
                   # char_emb_matrix = tf.concat([char_emb_pad,char_emb_matrix],0)

            char_context_emb = tf.nn.embedding_lookup(char_emb_matrix, context_char)
            char_question_emb = tf.nn.embedding_lookup(char_emb_matrix, question_char)

            with tf.variable_scope('char'):
                char_conv_context = char_conv(
                        char_context_emb,char_emb_dim,char_out,
                        filter_width,'VALID',
                        scope = 'char_context', keep_prob = cnn_keep_prob)
                if share_conv:
                    tf.get_variable_scope().reuse_variables()
                    char_conv_question = char_conv(
                            char_question_emb,char_emb_dim, char_out,
                            filter_width,'VALID', 
                            scope = 'char_context', keep_prob = cnn_keep_prob) 
                else:
                    char_conv_question = char_conv(
                              char_question_emb,char_emb_dem, self.char_out,
                            filter_width,'VALID', 
                            scope = 'char_question', keep_prob = cnn_keep_prob)
        return char_conv_context, char_conv_question


    def contextual_embedding_layer(self, xx, qq, reuse=None):
        ### contextual embedding layer
        if self.load_seo: vs="prepro"
        else: vs="Contextual_Embedding_Layer"
        with tf.variable_scope(vs, reuse=reuse) as scope:
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(
                    self.d_cell_fw, self.d_cell_bw, qq, 
                    self.con_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(axis=2, values=[fw_u, bw_u])
            if self.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, \
                                xx, self.con_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, \
                                xx, self.con_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
        return h, u

    
    def attention_flow_layer(self, h, u, reuse=None):
        if self.load_seo: vs="main"
        else: vs="Attention_Flow_Layer"
        with tf.device('/gpu:1'):
            with tf.variable_scope(vs, reuse=reuse) as scope:
                h_mask = self.x_mask
                u_mask = self.q_mask
                JX = tf.shape(h)[2]
                M = tf.shape(h)[1]
                JQ = tf.shape(u)[1]
                h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
                u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])

                if h_mask is None:
                    hu_mask = None
                else:
                    h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
                    u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
                    hu_mask = h_mask_aug & u_mask_aug
                if self.load_seo:
                    with tf.variable_scope("p0/bi_attention"):
                        u_logits = get_logits([h_aug, u_aug], None, True, wd=self.wd, mask=hu_mask,
                                              is_train=self.is_train, func=self.logit_func, scope='u_logits')  # [N, M, JX, JQ]
                else : 
                    u_logits = get_logits([h_aug, u_aug], None, True, wd=self.wd, mask=hu_mask,
                                          is_train=self.is_train, func=self.logit_func, scope='u_logits')  # [N, M, JX, JQ]
                u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
                h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
                h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

                # TODO: What is purpose of tensordict['a_u']? (and a_h)

                p0 = tf.concat(axis=3, values=[h, u_a, h*u_a, h*h_a])
            return p0
    
    def modeling_layer(self, p0, reuse=None):
        if self.load_seo: vs="main"
        else: vs="Modeling_Layer"
        with tf.variable_scope(vs, reuse=reuse) as scope:
            first_cell_fw = self.d_cell2_fw
            second_cell_fw = self.d_cell3_fw
            first_cell_bw = self.d_cell2_bw
            second_cell_bw = self.d_cell3_bw
            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell_fw, first_cell_bw, 
                                p0, self.con_len, 
                                dtype='float', scope='g0')  # [N, M, JX, 2d]
            g0 = tf.concat(axis=3, values=[fw_g0, bw_g0])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(second_cell_fw, second_cell_bw, 
                                g0, self.con_len, 
                                dtype='float', scope='g1')  # [N, M, JX, 2d]
            g1 = tf.concat(axis=3, values=[fw_g1, bw_g1])
            return g1

    def output_layer(self, p0, g1, reuse=None):
        if self.load_seo: vs="main"
        else: vs="Output_Layer"
        with tf.variable_scope(vs, reuse=reuse) as scope:
            N = tf.shape(p0)[0]
            M = 1
            JX = self.context_maxlen
            JQ = self.question_maxlen
            d = self.hidden_size
            logits = get_logits([g1, p0], d, True, wd=self.wd, 
                                input_keep_prob=self.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, 
                                func=self.answer_func, scope='logits1')
            a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), 
                            tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(  # [N, M, JX, 2d]
                                self.d_cell4_fw, self.d_cell4_bw, 
                                tf.concat(axis=3, values=[p0, g1, a1i, g1 * a1i]),
                                self.con_len, dtype='float', scope='g2')  
            
            g2 = tf.concat(axis=3, values=[fw_g2, bw_g2])
            logits2 = get_logits([g2, p0], d, True, 
                                wd=self.wd, input_keep_prob=self.input_keep_prob,
                                 mask=self.x_mask, is_train=self.is_train, 
                                 func=self.answer_func, scope='logits2')
            
            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)

            flat_logits = tf.Print(flat_logits, [flat_logits], "flat_logits : ")
            flat_logits2 = tf.Print(flat_logits2, [flat_logits2], "flat_logits2 : ")

            yp = tf.reshape(flat_yp, [-1, M, JX])
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])
            wyp = tf.nn.sigmoid(logits2)

            self.yp = yp
            self.yp2 = yp2
            self.wyp = wyp
        return flat_logits, flat_logits2

    def build_model(self):
        print("### Building a BiDAF model ###")

        xx, qq = self.init_bidaf()
        H, U = self.contextual_embedding_layer(xx, qq)
        print('# Contextual_Embedding_layer', H, U)
        G = self.attention_flow_layer(H, U)
        print('# Attention_Flow_layer', G)
        M = self.modeling_layer(G)
        print('# Modeling_layer', M)
        self.start_logits, self.end_logits = self.output_layer(G, M)
        print('# Output_layer', self.start_logits, self.end_logits)
        print(tf.global_variables())
        print(tf.trainable_variables())
        self.optimize_loss(self.start_logits, self.end_logits)


