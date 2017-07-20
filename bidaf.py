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
        # params hi hyunjae
        self.input_keep_prob = params['input_keep_prob']
        self.share_lstm_weights = params['share_lstm_weights']
        self.wd = params['wd']
        self.logit_func = params['logit_func']
        self.answer_func = params['answer_func']
        # Placeholders
        self.is_train = tf.placeholder('bool')
        
        super(BiDAF, self).__init__(params, initializer)

    #def character_embedding_layer():

    #def word_embedding_layer():
    
    def contextual_embedding_layer(self, xx, qq, reuse=None):
        ### contextual embedding layer
        with tf.variable_scope("Contextual_Embedding_Layer", reuse=reuse) as scope:
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
        with tf.variable_scope("Attention_Flow_Layer", reuse=reuse) as scope:
            JX = self.context_maxlen
            M = 1
            JQ = self.question_maxlen
            h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
            u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])

            # TODO: What is purpose of hu_mask?
            hu_mask = None

            u_logits = get_logits([h_aug, u_aug], None, True, wd=self.wd, mask=hu_mask,
                                  is_train=self.is_train, func=self.logit_func, scope='u_logits')  # [N, M, JX, JQ]
            u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
            h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
            h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

            # TODO: What is purpose of tensordict['a_u']? (and a_h)

            p0 = tf.concat(axis=3, values=[h, u_a, h*u_a, h*h_a])
        return p0
    
    def modeling_layer(self, p0, reuse=None):
        with tf.variable_scope("Modeling_Layer", reuse=reuse) as scope:
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
        with tf.variable_scope("Output_Layer", reuse=reuse) as scope:
            N = tf.shape(p0)[0]
            M = 1
            JX = self.context_maxlen
            JQ = self.question_maxlen
            d = self.dim_embed_word
            logits = get_logits([g1, p0], d, True, wd=self.wd, 
                                input_keep_prob=self.input_keep_prob,
                                is_train=self.is_train, 
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
                                 is_train=self.is_train, 
                                 func=self.answer_func, scope='logits2')
            
            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)

            yp = tf.reshape(flat_yp, [-1, M, JX])
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])
            wyp = tf.nn.sigmoid(logits2)

            print("yp 1,2 : ", yp, yp2)
            self.yp = yp
            self.yp2 = yp2
            self.wyp = wyp
        return flat_logits, flat_logits2

    def build_model(self):
        print("### Building a BiDAF model ###")
        self.cell_fw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.cell_bw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.d_cell_fw = SwitchableDropoutWrapper(self.cell_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell_bw = SwitchableDropoutWrapper(self.cell_bw, self.is_train,\
                    input_keep_prob=self.input_keep_prob)
        self.cell2_fw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.cell2_bw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.d_cell2_fw = SwitchableDropoutWrapper(self.cell2_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell2_bw = SwitchableDropoutWrapper(self.cell2_bw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.cell3_fw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.cell3_bw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.d_cell3_fw = SwitchableDropoutWrapper(self.cell3_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell3_bw = SwitchableDropoutWrapper(self.cell3_bw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.cell4_fw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.cell4_bw = BasicLSTMCell(self.dim_embed_word, state_is_tuple=True)
        self.d_cell4_fw = SwitchableDropoutWrapper(self.cell4_fw, self.is_train, \
                    input_keep_prob=self.input_keep_prob)
        self.d_cell4_bw = SwitchableDropoutWrapper(self.cell4_bw, self.is_train, \
                  input_keep_prob=self.input_keep_prob)

        # TODO: mask
        #tiled = tf.tile(tf.expand_dims(self.context_len, 1), [1, self.context_maxlen])
        #_range = tf.tile(tf.expand_dims(tf.range(0, self.context_maxlen), 1), 
        #        [1, self.context_len.get_shape().as_list()[0]])
        #print("before : ", tiled, _range)
        #_range = tf.traspose(_range)
        #print("after : ", tiled, _range)
        #self.x_mask = tf.less(_range, tiled)
        #self.x_mask = tf.Print(self.x_mask, [self.x_mask], "self.x_mask : ")
        #self.x_mask = tf.placeholder('bool', [None, None, None], name='x_mask')
        #self.q_mask = tf.placeholder('bool', [None, None], name='q_mask')
        
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
        
        self.con_len = tf.expand_dims(self.context_len, 1)
        #context_embed = tf.expand_dims(context_embed, 1)
        #context_embed = tf.Print(context_embed, [context_embed], "context_embed")
        xx = tf.expand_dims(context_embed, 1)
        #C = self.character_embedding_layer(context_embed, question_embed)
        #print('# Character_Embedding_layer', C)
        #X, Q = self.word_embedding_layer(context_embed, question_embed)
        #print('# Word_Embedding_layer', X, Q)
        H, U = self.contextual_embedding_layer(xx, question_embed)
        print('# Contextual_Embedding_layer', H, U)
        G = self.attention_flow_layer(H, U)
        print('# Attention_Flow_layer', G)
        M = self.modeling_layer(G)
        print('# Modeling_layer', M)
        self.start_logits, self.end_logits = self.output_layer(G, M)
        print('# Output_layer', self.start_logits, self.end_logits)

        self.optimize_loss(self.start_logits, self.end_logits)


