import tensorflow as tf
import sys

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        self.dim_perspective = params['dim_perspective']
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question):
        with tf.variable_scope('Filter_Layer') as scope:
            c_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(context), axis=-1), 1e-6))
            q_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(question), axis=-1), 1e-6))
            n_context = context / tf.expand_dims(c_norm, -1)
            n_question = question / tf.expand_dims(q_norm, -1)
            tr_question = tf.transpose(n_question, [0, 2, 1])
            similarity = tf.matmul(n_context, tr_question)
            max_similarity = tf.reduce_max(similarity, 2, keep_dims=True)
            # max_similarity = tf.Print(max_similarity, [similarity, max_similarity], 'max similarity', summarize=50)
            return tf.multiply(context, max_similarity)

    def representation_layer(self, inputs, length, max_length, scope=None):
        with tf.variable_scope('Representation_Layer/' + scope) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(inputs, self.dim_embed_word, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            # outputs = tf.Print(outputs, [outputs], 'bi output', summarize=20 * 31) 
            return outputs
    
    def matching_layer(self, context, question):
        with tf.variable_scope('Matching_Layer') as scope: 
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
              
                def cosine_dist(a, b):
                    a_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(a), axis=-1), 1e-6))
                    b_norm = tf.sqrt(tf.maximum(
                        tf.reduce_sum(tf.square(b), axis=-1), 1e-6))
                    result = tf.reduce_sum(tf.multiply(a, b), -1) / a_norm / b_norm
                    return result

                with tf.device('/gpu:1'):
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
                # result = tf.Print(result, [fw_indices, bw_indices], 'indices', summarize=10)
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
                full_fw, max_fw, mean_fw = tf.split(fw_matching, axis=3, num_or_size_splits=3)
                full_bw, max_bw, mean_bw = tf.split(bw_matching, axis=3, num_or_size_splits=3)
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
        with tf.variable_scope('Aggregation_Layer') as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.rnn_layer, self.rnn_dropout)
            r_inputs = rnn_reshape(inputs, self.dim_perspective * 6, max_length)
            # r_inputs = rnn_reshape(inputs, self.dim_rnn_cell * 2, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            print('\tinputs', inputs)
            print('\toutputs', outputs)
            return outputs

    def prediction_layer(self, inputs):
        with tf.variable_scope('Prediction_Layer') as scope:
            start_hidden = linear(inputs=inputs,
                output_dim=self.dim_hidden,
                activation=tf.nn.relu,
                dropout_rate=self.hidden_dropout,
                scope='Hidden_s')
            start_logits = linear(inputs=start_hidden,
                output_dim=1,
                scope='Output_s')
            start_logits = tf.reshape(start_logits, [-1, self.dim_output])

            end_hidden = linear(inputs=inputs,
                output_dim=self.dim_hidden,
                activation=tf.nn.relu,
                dropout_rate=self.hidden_dropout,
                scope='Hidden_e')
            end_logits = linear(inputs=end_hidden,
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
            # For skipping matching layer
            aggregates = context_filtered
            """
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
            aggregates = self.aggregation_layer(matchings, self.context_maxlen, self.context_len)
            # aggregates = self.aggregation_layer(context_rep, self.context_maxlen, self.context_len)
            print('# Aggregation_layer', aggregates)        

        with tf.device('/gpu:0'):
            start_logits, end_logits = self.prediction_layer(aggregates)
            # start_logits, end_logits = self.prediction_layer(context_rep)
            print('# Prediction_layer', start_logits, end_logits)

        return start_logits, end_logits
 
