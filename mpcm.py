import tensorflow as tf
import sys

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        self.dim_perspective = params['dim_perspective']
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question, reuse=None):
        with tf.variable_scope('Filter_Layer', reuse=reuse) as scope:
            c_norm = tf.sqrt(
                    tf.maximum(tf.reduce_sum(tf.square(context), axis=-1), 1e-6))
            q_norm = tf.sqrt(
                    tf.maximum(tf.reduce_sum(tf.square(question), axis=-1), 1e-6))
            n_context = context / tf.expand_dims(c_norm, -1)
            n_question = question / tf.expand_dims(q_norm, -1)
            tr_question = tf.transpose(n_question, [0, 2, 1])
            self.similarity = tf.matmul(n_context, tr_question)
            max_similarity = tf.reduce_max(self.similarity, 2, keep_dims=True)
            return tf.multiply(context, max_similarity)

    def representation_layer(self, inputs, length, max_length, scope=None, reuse=None):
        with tf.variable_scope('Representation_Layer/' + scope, reuse=reuse) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            outputs, state = bi_rnn_model(inputs, length, fw_cell, bw_cell)
            return outputs, state
    
    def matching_layer(self, context, question, reuse=None):
        with tf.variable_scope('Matching_Layer', reuse=reuse) as scope: 
            def matching_function(w, c, q):
                with tf.device('/gpu:0'):
                    # [B, C, 2H] => [B, C, 1, 2H] => [C, B, 3L, 2H]
                    c_e = tf.expand_dims(c, 2)
                    c_w = tf.multiply(c_e, w)
                    c_t = tf.transpose(c_w, [1, 0, 2, 3])
                    cf, cb = tf.split(c_t, num_or_size_splits=2, axis=3)

                    # [B, Q, 2H] => [B, Q, 1, 2H] => [Q, B, 3L, 2H]
                    q_e = tf.expand_dims(q, 2)
                    q_w = tf.multiply(q_e, w)
                    q_t = tf.transpose(q_w, [1, 0, 2, 3])
                    qf, qb = tf.split(q_t, num_or_size_splits=2, axis=3)

                    # Normalize context and question
                    cf_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(cf), axis=-1, keep_dims=True), 1e-6))
                    cb_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(cb), axis=-1, keep_dims=True), 1e-6))
                    cf /= cf_norm
                    cb /= cb_norm
                    qf_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(qf), axis=-1, keep_dims=True), 1e-6))
                    qb_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(qb), axis=-1, keep_dims=True), 1e-6))
                    qf /= qf_norm
                    qb /= qb_norm
              
                def cosine_dist(a, b):
                    result = tf.reduce_sum(tf.multiply(a, b), -1)
                    return result

                with tf.device('/gpu:0'):
                    # [C, B, 3L, H] X [Q, B, 3L, H] => [C, Q, B, 3L]
                    q_shape = tf.shape(qf)
                    init = tf.zeros([q_shape[0], q_shape[1], q_shape[2]])
                    fw_result = tf.scan(lambda a, x: cosine_dist(x, qf), cf, init)
                    bw_result = tf.scan(lambda a, x: cosine_dist(x, qb), cb, init)
                
                with tf.device('/gpu:0'):
                    # [C, Q, B, 3L] => [B, C, Q, 3L]
                    fw_tr = tf.transpose(fw_result, [2, 0, 1, 3])
                    bw_tr = tf.transpose(bw_result, [2, 0, 1, 3])
                    print('\tmatching function', fw_tr)

                return fw_tr, bw_tr

            def full_matching(fw, bw):
                batch_size = tf.shape(fw)[0]
                batch_index = tf.reshape(tf.tile(
                    tf.expand_dims(tf.range(0, batch_size), 1), 
                    [1, self.context_maxlen]), [-1])
                context_index = tf.reshape(tf.tile(
                    tf.expand_dims(tf.range(0, self.context_maxlen), 0),
                    [batch_size, 1]), [-1])
                question_index = tf.reshape(tf.tile(
                    tf.expand_dims(self.question_len - 1, 1),
                    [1, self.context_maxlen]), [-1])
                fw_indices = tf.concat([tf.expand_dims(batch_index, 1),
                    tf.expand_dims(context_index, 1),
                    tf.expand_dims(question_index, 1)], axis=1)
                bw_indices = tf.concat([tf.expand_dims(batch_index, 1),
                    tf.expand_dims(context_index, 1),
                    tf.expand_dims(tf.zeros([batch_size * self.context_maxlen], 
                        dtype=tf.int32), 1)], axis=1)

                gathered_fw = tf.reshape(tf.gather_nd(fw, fw_indices), 
                        [-1, self.context_maxlen, self.dim_perspective])
                gathered_bw = tf.reshape(tf.gather_nd(bw, bw_indices),
                        [-1, self.context_maxlen, self.dim_perspective])
                
                result = tf.concat([gathered_fw, gathered_bw], axis=2)
                print('\tfull matching', result)
                return result

            def max_matching(fw, bw):
                gathered_fw = tf.reduce_max(fw, axis=2)
                gathered_bw = tf.reduce_max(bw, axis=2)
                
                result = tf.concat([gathered_fw, gathered_bw], axis=2)
                print('\tmax matching', result)
                return result

            def mean_matching(fw, bw):
                gathered_fw = tf.reduce_mean(fw, axis=2)
                gathered_bw = tf.reduce_mean(bw, axis=2)
                
                result = tf.concat([gathered_fw, gathered_bw], axis=2)
                print('\tmean matching', result)
                return result
           
            w_matching = tf.get_variable('w_matching', 
                    [self.dim_perspective * 3, self.dim_rnn_cell * 2],
                    initializer=tf.random_normal_initializer(), dtype=tf.float32)
            fw_matching, bw_matching = matching_function(w_matching, context, question)
            
            with tf.device('/gpu:0'):
                full_fw, max_fw, mean_fw = tf.split(
                        fw_matching, axis=3, num_or_size_splits=3)
                full_bw, max_bw, mean_bw = tf.split(
                        bw_matching, axis=3, num_or_size_splits=3)
                full_result = full_matching(full_fw, full_bw)
                max_result = max_matching(max_fw, max_bw)
                mean_result = mean_matching(mean_fw, mean_bw)
            
                # full_result = tf.Print(full_result, [full_result], 'f', summarize=10)
                result = tf.concat([full_result, max_result, mean_result], axis=2)
                print('\tmatching_result', result)
            
            return result

    def aggregation_layer(self, inputs, max_length, length, reuse=None):
        with tf.variable_scope('Aggregation_Layer', reuse=reuse) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            outputs, _ = bi_rnn_model(inputs, length, fw_cell, bw_cell)
            
            print('\tinputs', inputs)
            print('\toutputs', outputs)
            return outputs

    def prediction_layer(self, inputs, reuse=None):
        with tf.variable_scope('Prediction_Layer', reuse=reuse) as scope:
            """
            start_hidden = linear(inputs=inputs,
                output_dim=self.dim_hidden,
                activation=tf.nn.relu,
                dropout_rate=self.hidden_dropout,
                scope='Hidden_s')
            """
            start_logits = linear(inputs=inputs,
                output_dim=1,
                scope='Output_s')
            start_logits = tf.reshape(start_logits, [-1, self.dim_output])
            
            """
            end_hidden = linear(inputs=inputs,
                output_dim=self.dim_hidden,
                activation=tf.nn.relu,
                dropout_rate=self.hidden_dropout,
                scope='Hidden_e')
            """
            end_logits = linear(inputs=inputs,
                output_dim=1,
                scope='Output_e')
            end_logits = tf.reshape(end_logits, [-1, self.dim_output])
            
            # Masking start, end logits
            start_logits = tf.multiply(start_logits, self.context_mask)
            end_logits = tf.multiply(end_logits, self.context_mask)
            return start_logits, end_logits
    
    def char_conv(self, inputs, emb_dim, output_dim, 
            filter_width, padding,keep_prob = 1.0,scope = None):
        with tf.variable_scope('conv' or scope):
            num_channels = emb_dim
            conv_filter = tf.get_variable("filter", 
                    shape=[1,filter_width, num_channels, output_dim], 
                    dtype=tf.float32)
            bias = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32)
            strides = [1,1,1,1]
            inputs = dropout(inputs, keep_prob)
            conv = tf.nn.conv2d(inputs, conv_filter, strides, padding) + bias
            conv_output = tf.reduce_max(tf.nn.relu(conv),2)
            return conv_output


    def char_emb_layer(self, context_char, question_char, char_size, 
            char_emb_dim, char_out, filter_width, cnn_keep_prob, share_conv):
        with tf.variable_scope("char"): 
            char_emb_matrix = tf.get_variable(
                    "char_emb_matrix",shape = [(char_size),char_emb_dim],
                    dtype = tf.float32, trainable = True)
               # char_emb_pad = tf.constant(([[0.0]*self.char_emb_dim]),
               #    dtype = tf.float32)
               # char_emb_matrix = tf.concat([char_emb_pad,char_emb_matrix],0)

            char_context_emb = tf.nn.embedding_lookup(char_emb_matrix, context_char)
            char_question_emb = tf.nn.embedding_lookup(char_emb_matrix, question_char)

            with tf.variable_scope('conv'):
                char_conv_context = self.char_conv(
                        char_context_emb,char_emb_dim,char_out,
                        filter_width,'VALID',
                        scope = 'char_context', keep_prob = cnn_keep_prob)
                if share_conv:
                    tf.get_variable_scope().reuse_variables()
                    char_conv_question = self.char_conv(
                            char_question_emb,char_emb_dim, char_out,
                            filter_width,'VALID', 
                            scope = 'char_context', keep_prob = cnn_keep_prob) 
                else:
                    char_conv_question = self.char_conv(
                              char_question_emb,char_emb_dem, self.char_out,
                            filter_width,'VALID', 
                            scope = 'char_question', keep_prob = cnn_keep_prob)
        return char_conv_context, char_conv_question

    def build_model(self):
        print("### Building MPCM model ###")

        with tf.device('/gpu:0'):
            context_embed = embedding_lookup(
                    inputs=self.context,
                    voca_size=self.voca_size,
                    embedding_dim=self.dim_embed_word, 
                    initializer=self.initializer, 
                    trainable=self.embed_trainable,
                    reuse=True, scope='Word')
            
            question_embed = embedding_lookup(
                    inputs=self.question,
                    voca_size=self.voca_size,
                    embedding_dim=self.dim_embed_word,
                    initializer=self.initializer,
                    trainable=self.embed_trainable,
                    reuse=True, scope='Word')
            print(context_embed)
            print(question_embed)
            
            char_context_embed, char_question_embed = self.char_emb_layer(
                    self.context_char, self.question_char,
                    self.char_size, self.char_emb_dim, 
                    self.char_out, self.filter_width, 
                    self.cnn_keep_prob, self.share_conv)

            print(char_context_embed)
            print(char_question_embed)
            
            context_embed_input = dropout(tf.concat(
                [context_embed, char_context_embed],2), self.embed_dropout)
            question_embed_input = dropout(tf.concat(
                [question_embed, char_question_embed],2),self.embed_dropout)
            
            print(context_embed_input)
            print(question_embed_input)
                         
            context_filtered = self.filter_layer(
                    context_embed_input, question_embed_input)
            # context_filtered = dropout(context_filtered, self.embed_dropout)
            context_filtered = self.apply_mask(context_filtered, self.context_mask)
            print('# Filter_layer', context_filtered)
            
            context_rep, _ = self.representation_layer(context_filtered, 
                    self.context_len, self.context_maxlen, scope='Context')
            # context_rep = dropout(context_rep, self.embed_dropout)
            context_rep = self.apply_mask(context_rep, self.context_mask)

            question_rep, _ = self.representation_layer(question_embed, 
                    self.question_len, self.question_maxlen, scope='Question')
            # question_rep = dropout(question_rep, self.embed_dropout)
            question_rep = self.apply_mask(question_rep, self.question_mask)
            print('# Representation_layer', context_rep, question_rep)

        matchings = self.matching_layer(context_rep, question_rep)
        # matchings = dropout(matchings, self.embed_dropout)
        matchings = self.apply_mask(matchings, self.context_mask)
        print('# Matching_layer', matchings)

        with tf.device('/gpu:0'):
            aggregates = self.aggregation_layer(
                    matchings, self.context_maxlen, self.context_len)
            # aggregates = dropout(aggregates, self.embed_dropout)
            aggregates = self.apply_mask(aggregates, self.context_mask)
            print('# Aggregation_layer', aggregates)        

        with tf.device('/gpu:0'):
            self.start_logits, self.end_logits = self.prediction_layer(aggregates)
            print('# Prediction_layer', self.start_logits, self.end_logits)
        

        self.optimize_loss(self.start_logits, self.end_logits, self.softmax_dropout)
 
