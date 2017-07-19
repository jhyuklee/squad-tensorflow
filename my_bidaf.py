import tensorflow as tf
import sys
import time
from model import Basic
from ops import *

def bidirectional_LSTM(hidden_size, keep_prob, X, time_step, X_len, output_dim):
    print("  - call -> bidirectional_LSTM()   ")
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    fw_drop = tf.contrib.rnn.DropoutWrapper(
                lstm_fw_cell, output_keep_prob=keep_prob)
    bw_drop = tf.contrib.rnn.DropoutWrapper(
                lstm_bw_cell, output_keep_prob=keep_prob)

    print("X, time_step, X_len : ", X, time_step, X_len)
    outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_drop, bw_drop, X, sequence_length=X_len, dtype=tf.float32)
    
    stacked_output = tf.stack(outputs)
    # outputs : shape=(seq_size, batch_size, 2*lstm_size)
    stacked_output = tf.transpose(stacked_output, [1,0,2])  
    # outputs : shape=(batch_size, seq_size, 2*lstm_size)

    X = tf.stack(X)
    # (time step, ?, d)
    W = tf.Variable(tf.random_normal([2*hidden_size, int(X.get_shape().as_list()[-1]*output_dim)]))
    b = tf.Variable(tf.constant(0.0, shape = [int(X.get_shape().as_list()[-1]*output_dim)]))

    reshaped_output = tf.reshape(stacked_output, [-1, 2*hidden_size])
    # (batch * time step , 2*lstm_size)
    reshaped_output = tf.matmul(reshaped_output, W) + b
    # (?, output_dim * d)

    Y = tf.reshape(reshaped_output, [-1, time_step, int(X.get_shape().as_list()[-1]*output_dim)])
    # (?, time step, output_dim * d)
    Y = tf.transpose(Y, [0, 2, 1])
    # (?, output_dim * d, time step)

    return Y


class My_BiDAF(Basic):
    def __init__(self, params, initializer):
        super(My_BiDAF, self).__init__(params, initializer)        


    def contextual_embedding_layer(self, X, Q):
        with tf.variable_scope("Contextual_embeding_layer") as v_scope:
            for switch in [("context", X, self.context_maxlen, self.context_len), \
                    ("question", Q, self.question_maxlen, self.question_len)]:
                with tf.variable_scope(switch[0]) as v_scope:
                    now = time.localtime()
                    print("*** Contextual embeding layer - {} ({}:{}:{})".format(switch[0], now.tm_hour, now.tm_min, now.tm_sec))
                    if switch[0] == "context":
                        H = bidirectional_LSTM(self.dim_hidden, self.rnn_dropout, switch[1], switch[2], switch[3], 2)
                        # (?, context time step, 2d)
                    else:
                        U = bidirectional_LSTM(self.dim_hidden, self.rnn_dropout, switch[1], switch[2], switch[3], 2)
                        # (?, question time step, 2d)

            return H, U
            

    
    def attention_flow_layer(self, H, U):	
        with tf.variable_scope("Attention_flow_layer") as v_scope:
            now = time.localtime()
            print("**** Attention flow layer ({}:{}:{})".format(
                            now.tm_hour, now.tm_min, now.tm_sec))
            B = H.get_shape().as_list()[0]  # batch size
            T = H.get_shape().as_list()[-1] # len of context
            J = U.get_shape().as_list()[-1] # len of question
            D = H.get_shape().as_list()[1]  # dimension
           
            H_trans = tf.transpose(H, [0, 2, 1]) # (B, T, D)
            U_trans = tf.transpose(U, [0, 2, 1]) # (B, J, D)
            HH = tf.expand_dims(H_trans, 2)     # (B, T, 1, D)
            HH = tf.tile(HH, [1, 1, J, 1])      # (B, T, J, D)
            UU = tf.expand_dims(U_trans, 1)     # (B, 1, J, D)
            H_mul_U = tf.multiply(HH, UU)       # (B, T, J, D)
            H_mul_U = tf.reshape(H_mul_U, [-1, T*J, D]) # (B, T*J, D)

            HHH = tf.reshape(HH, [-1, T*J, D])
            UUU = tf.tile(U_trans, [1, T, 1])

            concat = tf.concat([HHH, UUU, H_mul_U], 2) # (B, T*J, 3D)

            w_s = tf.Variable(tf.random_normal([3*D, 1]))

            reshape = tf.reshape(concat, [-1, 3*D]) # (B*T*J, 3D)
            alpha = tf.matmul(reshape, w_s)         # (B*T*J, 1)

            S = tf.reshape(alpha, [-1, T, J])       # (B, T, J)
            
            a = tf.nn.softmax(S, -1)
            # (?, context time step, que time step)
            print('a : ',a)

            attended_U = tf.matmul(U, tf.transpose(a, [0, 2, 1]))
            # (?, 2d, context time step)
            print('attended_U : ', attended_U)

            b = tf.nn.softmax(tf.reduce_max(S, 2), -1)
            # (?, context time step)
            print('b : ', b)

            attended_H = tf.multiply(H, tf.expand_dims(b, 1))
            print('attended_H : ', attended_H)

            h_el_mul_att_u = tf.multiply(H, attended_U)
            h_el_mul_att_h = tf.multiply(H, attended_H)
            
            print("hOu~ : ", h_el_mul_att_u)
            print("hOh~ : ", h_el_mul_att_h)

            G = tf.concat([H, attended_U, h_el_mul_att_u, h_el_mul_att_h], axis=1)
            
            return G


    def modeling_layer(self, G):
        with tf.variable_scope("Modeling_layer") as v_scope:
            now = time.localtime()
            print("***** Modeling layer ({}:{}:{})".format(
                        now.tm_hour, now.tm_min, now.tm_sec))
            G = tf.unstack(G, self.context_maxlen, 2)
            # (batch, 8d) * self.context_maxlen
                                    
            M = bidirectional_LSTM(self.dim_hidden, self.rnn_dropout, G, 
                    self.context_maxlen, self.context_len, 1/4)
            # (?, question time step, 2d)

            return M

    def output_layer(self, G, M):
        with tf.variable_scope("Output_layer") as v_scope:
            now = time.localtime()
            print("****** Output layer ({}:{}:{})".format(
                        now.tm_hour, now.tm_min, now.tm_sec))
            G_M = tf.concat([G, M], axis=1)
            #G_M = M
            print("G_M : ", G_M)
            # (?, 10d, time step)
            
            w_p1 = tf.Variable(tf.random_normal([5*M.get_shape().as_list()[1]]))
            print("w_p1 : ", w_p1)
            # (?, 10d)

            w_p1 = tf.expand_dims(w_p1, 1)
            w_p1 = tf.expand_dims(w_p1, 0)
            p1 = tf.reduce_sum(tf.multiply(w_p1, G_M), 1)
            print("p1 : ", p1)
            # (?, time step)

            M = tf.unstack(M, self.context_maxlen, 2)
            # (batch_size, embedding_size) * time_step_size
            M2 = bidirectional_LSTM(self.dim_hidden, self.rnn_dropout, M, 
                    self.context_maxlen, self.context_len, 1)
            # (?, 2d, time step)

            G_M2 = tf.concat([G, M2], axis=1)
            print("G_M2 : ", G_M2)
            # (?, 10d, time step)

            M = tf.stack(M)
            print('M after stack : ', M)
            w_p2 = tf.Variable(tf.random_normal([5*M.get_shape().as_list()[-1]]))
            print("w_p2 : ", w_p2)
            # (?, 10d, time step)

            w_p2 = tf.expand_dims(w_p2, 1)
            w_p2 = tf.expand_dims(w_p2, 0)
            p2 = tf.reduce_sum(tf.multiply(w_p2, G_M2), 1)
            #p2 = tf.Print(p2, [p2], "p2 : ")
            print("p2 : ", p2)
            # (?, time step)

            return p1, p2

    def build_model(self):
        print("### Building BiDAF model ###")
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
        
        H, U = self.contextual_embedding_layer(context_embed, question_embed)
        print('# Contextual_Embedding_layer', H, U)
        G = self.attention_flow_layer(H, U)
        print('# Attention_Flow_layer', G)
        M = self.modeling_layer(G)
        print('# Modeling_layer', M)
        self.start_logits, self.end_logits = self.output_layer(G, M)
        print('# Output_layer', self.start_logits, self.end_logits)

        self.optimize_loss(self.start_logits, self.end_logits)



