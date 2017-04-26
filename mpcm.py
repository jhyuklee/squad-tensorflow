import tensorflow as tf
import sys

from model import Basic
from ops import *


class MPCM(Basic):
    def __init__(self, params, initializer):
        self.max_perspective = params['max_perspective']
        super(MPCM, self).__init__(params, initializer)

    def filter_layer(self, context, question):
        c_norm = tf.norm(context, axis=2, keep_dims=True)
        q_norm = tf.norm(question, axis=2, keep_dims=True)
        n_context = context / c_norm
        n_question = question / q_norm
        tr_question = tf.transpose(n_question, [0, 2, 1])
        similarity = tf.matmul(n_context, tr_question)
        max_similarity = tf.reduce_max(similarity, 2, keep_dims=True)
        return tf.multiply(context, max_similarity)

    def representation_layer(self, inputs, length, max_length, scope=None):
        with tf.variable_scope('Representation/' + scope) as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            r_inputs = rnn_reshape(inputs, self.dim_embed_word, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            # TODO: Gather by length

            return outputs
    
    def matching_layer(self, context, question):
        fw_context, bw_context = tf.split(context, num_or_size_splits=2, axis=2)
        fw_question, bw_question = tf.split(question, num_or_size_splits=2, axis=2)
        
        def matching_function(v1, v2, W):
            # TODO: Normalize vectors
            cos_d = tf.scan(lambda a, W_k: (W_k * v1)*(W_k * v2), W)
            return tf.reduce_sum(cos_d, axis=1)

        W_fw = tf.get_variable('W_fw', [self.max_perspective, self.dim_rnn_cell],
                initializer=tf.random_normal_initializer())
        W_bw = tf.get_variable('W_bw', [self.max_perspective, self.dim_rnn_cell],
                initializer=tf.random_normal_initializer())
        
        def run_matching(single_context, single_question, context_len, question_len, fw=True):
            context_group = tf.unstack(single_context, self.context_maxlen)
            question_group = tf.unstack(single_question, self.question_maxlen)
            matching_list = []
            for ct_idx, ct in enumerate(context_group):
                for qu_idx, qu in enumerate(question_group):
                    init = tf.zeros([self.max_perspective])
                    W = W_fw if fw else W_bw
                    ct_const = tf.constant(ct_idx)
                    qu_const = tf.constant(qu_idx)
                    matching_result = tf.cond(tf.less(qu_const, question_len),
                        lambda: tf.cond(tf.less(ct_const, context_len), lambda: matching_function(ct, qu, W),
                        lambda: init), lambda: init)
                    matching_list.append(matching_result)

                _progress = '\r\t processing %d/%d' % (ct_idx, len(context_group))
                sys.stdout.write(_progress)
                sys.stdout.flush()
                # if ct_idx >= 9:
                #     break
            print()
            return tf.stack(matching_list)

        # TODO: use tf.cond for batch
        init = tf.zeros([self.context_maxlen * self.question_maxlen, self.max_perspective])
        fw_matching_total = tf.scan(lambda a, w: run_matching(w[0], w[1], w[2], w[3], True), 
                (fw_context, fw_question, self.context_len, self.question_len), init)
        print('\t', 'fw matching', fw_matching_total)
        bw_matching_total = tf.scan(lambda a, w: run_matching(w[0], w[1], w[2], w[3], False), 
                (bw_context, bw_question, self.context_len, self.question_len), init)
        print('\t', 'bw matching', bw_matching_total)
        
        def full_matching(fw_seq, bw_seq, length):
            print('\t', 'Full matching')
            """
            fw, bw_seq: [batch, question_length, context_length, perspective]
            length: [batch]
            """
            batch_size = tf.shape(fw_seq)[0]
            print('\t', 'before', fw_seq, bw_seq, length)
            indices = tf.concat(axis=1,
                    values=[tf.expand_dims(tf.range(0, batch_size), 1),
                    tf.expand_dims(length-1, 1)])
            last_gathered = tf.gather_nd(fw_seq, indices)
            print('\t', 'last gathered', last_gathered)
            
            indices = tf.concat(axis=1,
                    values=[tf.expand_dims(tf.range(0, batch_size), 1),
                    tf.cast(tf.zeros([batch_size, 1]), tf.int32)])
            first_gathered = tf.gather_nd(bw_seq, indices)
            print('\t', 'first gathered', first_gathered)
            
            result = tf.concat(axis=2, values=[last_gathered, first_gathered])
            return result

        def max_matching(sequence, length):
            # TODO: gather maximum between 0 ~ max in sequence
            return sequence

        def mean_matching(sequence, length):
            # TODO: gather mean of 0 ~ max in sequence
            return sequence

        # fw_full, fw_max, fw_mean = tf.split(fw_matching_total, num_or_size_splits=3, axis=2)
        # bw_full, bw_max, bw_mean = tf.split(bw_matching_total, num_or_size_splits=3, axis=2)
        fw_full = fw_matching_total
        bw_full = bw_matching_total
        
        c_len = self.context_maxlen
        # c_len = 10
        q_len = self.question_maxlen
        fw_full = tf.transpose(tf.reshape(fw_full, [-1, c_len, q_len, self.max_perspective]), 
                [0, 2, 1, 3])
        bw_full = tf.transpose(tf.reshape(bw_full, [-1, c_len, q_len, self.max_perspective]), 
                [0, 2, 1, 3])
        full_result = full_matching(fw_full, bw_full, self.question_len)

        return full_result

    def aggregation_layer(self, inputs, max_length, length):
        # max_length = 10 # For test
        with tf.variable_scope('Aggregation') as scope:
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            r_inputs = rnn_reshape(inputs, self.max_perspective * 2, max_length)
            outputs = bi_rnn_model(r_inputs, length, fw_cell, bw_cell)
            print('\t', 'inputs', inputs)
            print('\t', 'outputs', outputs)
            #TODO: gather only sentence boundary vectors => mask?
            return tf.reshape(outputs, [-1, max_length * self.dim_rnn_cell * 2])

    def prediction_layer(self, inputs):
        start_logits = linear(inputs=inputs,
            output_dim=self.dim_output, 
            scope='Output_s')

        end_logits = linear(inputs=inputs,
            output_dim=self.dim_output, 
            scope='Output_e')

        return start_logits, end_logits

    def build_model(self):
        print("### Building MPCM model ###")
         
        context_embed = embedding_lookup(
                inputs=self.context,
                voca_size=self.dim_word,
                embedding_dim=self.dim_embed_word, 
                initializer=self.initializer, 
                trainable=self.embed_trainable, scope='Word')

        question_embed = embedding_lookup(
                inputs=self.question,
                voca_size=self.dim_word,
                embedding_dim=self.dim_embed_word,
                initializer=self.initializer,
                trainable=self.embed_trainable,
                reuse=True, scope='Word')

        context_filtered = self.filter_layer(context_embed, question_embed)
        print('# Filter_layer', context_filtered)

        context_rep = self.representation_layer(context_filtered, self.context_len,
                self.context_maxlen, scope='Context')
        question_rep = self.representation_layer(question_embed, self.question_len,
                self.question_maxlen, scope='Question')
        print('# Representation_layer', context_rep, question_rep)

        matching_vectors = self.matching_layer(context_rep, question_rep)
        print('# Matching_layer', matching_vectors)

        aggregation = self.aggregation_layer(matching_vectors, self.context_maxlen, self.context_len)
        print('# Aggregation_layer', aggregation)
 
        self.start_logits, self.end_logits = self.prediction_layer(aggregation)
        print('# Prediction_layer', self.start_logits, self.end_logits)
 
