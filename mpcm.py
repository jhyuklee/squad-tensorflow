import tensorflow as tf
import sys

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        self.dim_perspective = params['dim_perspective']
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question):
        c_norm = tf.norm(context, axis=2, keep_dims=True)
        q_norm = tf.norm(question, axis=2, keep_dims=True)
        n_context = context / (c_norm + tf.constant(1e-10))
        n_question = question / (q_norm + tf.constant(1e-10))
        tr_question = tf.transpose(n_question, [0, 2, 1])
        similarity = tf.matmul(n_context, tr_question)
        max_similarity = tf.reduce_max(similarity, 2, keep_dims=True)
        return tf.multiply(context, max_similarity)

    def representation_layer(self, inputs, length, max_length, scope=None):
        with tf.variable_scope('Representation/' + scope) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(inputs, self.dim_embed_word, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            return outputs
    
    def matching_layer(self, context, question):

        def test_function(w, c, q):
            # [B, C, 2H] => [C, B, 2H]
            c_tr = tf.transpose(c, [1, 0, 2])

            # [B, Q, 2H] => [Q, B, 2H] => [Q, B, 3L, 2H] => [Q, B, 6L, H]
            q_tr = tf.transpose(q, [1, 0, 2])
            q_tiled = tf.tile(tf.expand_dims(q_tr, 2), [1, 1, self.dim_perspective * 3, 1])
            q_fb = tf.reshape(q_tiled,
                    [self.question_maxlen, -1, self.dim_perspective * 6, self.dim_rnn_cell])
            q_w = tf.multiply(q_fb, w)
            
            def do_matching(a, b, l):
                # [B, 2H] => [B, 3L, 2H] => [B, 6L, H]
                a_tiled = tf.tile(tf.expand_dims(a, 1), [1, self.dim_perspective * 3, 1])
                a_fb = tf.reshape(a_tiled, [-1, self.dim_perspective * 6, self.dim_rnn_cell])
                a_w = tf.multiply(a_fb, l)

                # Normalize
                a_norm = tf.norm(a_w, axis=2, keep_dims=True)
                b_norm = tf.norm(b, axis=3, keep_dims=True)
                a_w = a_w / (a_norm + tf.constant(1e-10))
                b = b / (b_norm + tf.constant(1e-10))
                
                # [Q, B, 6L]
                return tf.reduce_sum(tf.multiply(a_w, b), 3)

            # [C, B, 2H] X [Q, B, 6L, H] => [C, Q, B, 6L]
            q_shape = tf.shape(q_tr)
            init = tf.zeros([q_shape[0], q_shape[1], self.dim_perspective * 6])
            w_result = tf.scan(lambda a, x: do_matching(x, q_w, w), c_tr, init)
            print('w_result', w_result)
            
            # [C, Q, B, 6L] => [B, C, Q, 6L]
            w_tr = tf.transpose(w_result, [2, 0, 1, 3])
            print('\tmatching function', w_tr)
            return w_tr
        
        def matching_function(w, c, q):
            with tf.device('/gpu:0'):
                # [B, C, 2H] => [B, C, 3L, 2H] => [B, C, 6L, H] => [C, B, 6L, H]
                c_tiled = tf.tile(tf.expand_dims(c, 2), [1, 1, self.dim_perspective * 3, 1])
                c_fb = tf.reshape(c_tiled, 
                        [-1, self.context_maxlen, self.dim_perspective * 6, self.dim_rnn_cell])
                c_w = tf.multiply(c_fb, w)
                c_t = tf.transpose(c_w, [1, 0, 2, 3])

                # [B, Q, 2H] => [B, Q, 3L, 2H] => [B, Q, 6L, H] => [Q, B, 6L, H]
                q_tiled = tf.tile(tf.expand_dims(q, 2), [1, 1, self.dim_perspective * 3, 1])
                q_fb = tf.reshape(q_tiled,
                        [-1, self.question_maxlen, self.dim_perspective * 6, self.dim_rnn_cell])
                q_w = tf.multiply(q_fb, w)
                q_t = tf.transpose(q_w, [1, 0, 2, 3])
            
            def reduce_norm_mul(a, b):
                a_norm = tf.norm(a, axis=2, keep_dims=True)
                b_norm = tf.norm(b, axis=3, keep_dims=True)
                a = a / (a_norm + tf.constant(1e-10))
                b = b / (b_norm + tf.constant(1e-10))
                result = tf.reduce_sum(tf.multiply(a, b), 3)
                return result

            with tf.device('/gpu:1'):
                # [C, B, 6L, H] X [Q, B, 6L, H] => [C, Q, B, 6L]
                q_shape = tf.shape(q_w)
                init = tf.zeros([q_shape[0], q_shape[1], q_shape[2]])
                w_result = tf.scan(lambda a, x: reduce_norm_mul(x, q_t), c_t, init)
            
            with tf.device('/gpu:0'):
                # [C, Q, B, 6L] => [B, C, Q, 6L]
                w_tr = tf.transpose(w_result, [2, 0, 1, 3])
                print('\tmatching function', w_tr)

            return w_tr

        def full_matching(fw, bw):
            batch_size = tf.shape(fw)[0]
            batch_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), 
                    [1, self.context_maxlen]), [-1])
            context_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, self.context_maxlen), 0),
                    [batch_size, 1]), [-1])
            question_index = tf.reshape(tf.tile(tf.expand_dims(self.question_len - 1, 1),
                    [1, self.context_maxlen]), [-1])
            fw_indices = tf.concat([tf.expand_dims(batch_index, 1),
                tf.expand_dims(context_index, 1),
                tf.expand_dims(question_index, 1)], axis=1)
            bw_indices = tf.concat([tf.expand_dims(batch_index, 1),
                tf.expand_dims(context_index, 1),
                tf.expand_dims(tf.zeros([batch_size * self.context_maxlen], dtype=tf.int32), 1)], axis=1)

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
       
        w_matching = tf.get_variable('w_matching', [self.dim_perspective * 6, self.dim_rnn_cell],
                initializer=tf.random_normal_initializer(), dtype=tf.float32)
        matching_result = matching_function(w_matching, context, question)
        
        with tf.device('/gpu:0'):
            full_fw, full_bw, max_fw, max_bw, mean_fw, mean_bw = tf.split(matching_result, axis=3,
                    num_or_size_splits=6)
            full_result = full_matching(full_fw, full_bw)
            max_result = max_matching(max_fw, max_bw)
            mean_result = mean_matching(mean_fw, mean_bw)
        
            # full_result = tf.Print(full_result, [full_result], 'full', summarize=10)
            # max_result = tf.Print(max_result, [max_result], 'max', summarize=10)
            # mean_result = tf.Print(mean_result, [mean_result], 'mean', summarize=10)

            result = tf.concat([full_result, max_result, mean_result], axis=2)
            print('\tmatching_result', result)
        
        return result

    def aggregation_layer(self, inputs, max_length, length):
        with tf.variable_scope('Aggregation') as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            # r_inputs = rnn_reshape(inputs, self.dim_perspective * 6, max_length)
            r_inputs = rnn_reshape(inputs, self.dim_rnn_cell, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            print('\tinputs', inputs)
            print('\toutputs', outputs)
            return outputs

    def prediction_layer(self, inputs):
        start_logits = linear(inputs=inputs,
            output_dim=1, 
            scope='Output_s')
        start_logits = tf.reshape(start_logits, [-1, self.dim_output])

        end_logits = linear(inputs=inputs,
            output_dim=1, 
            scope='Output_e')
        end_logits = tf.reshape(end_logits, [-1, self.dim_output])
        
        return start_logits, end_logits

    def build_model(self):
        print("### Building MPCM model ###")

        with tf.device('/gpu:0'):
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

            context_filtered = self.filter_layer(context_embed, question_embed)
            print('# Filter_layer', context_filtered)
            
            """
            # For skipping rep layer
            self.dim_rnn_cell = int(self.dim_embed_word / 2)
            aggregates = self.matching_layer(context_filtered, question_embed)
            """

            context_rep = self.representation_layer(context_filtered, self.context_len,
                    self.context_maxlen, scope='Context')
            question_rep = self.representation_layer(question_embed, self.question_len,
                    self.question_maxlen, scope='Question')
            print('# Representation_layer', context_rep, question_rep)

        matchings = self.matching_layer(context_rep, question_rep)
        print('# Matching_layer', matchings)

        with tf.device('/gpu:0'):
            aggregates = self.aggregation_layer(context_filtered, self.context_maxlen, self.context_len)
            print('# Aggregation_layer', aggregates)
                
            start_logits, end_logits = self.prediction_layer(aggregates)
            print('# Prediction_layer', start_logits, end_logits)

        return start_logits, end_logits
 
